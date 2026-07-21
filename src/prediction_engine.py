import pandas as pd
import numpy as np
from src.descriptor_engine import DescriptorEngine
from src.utils import setup_logger

logger = setup_logger(__name__)

class PredictionEngine:
    """Predicts properties for untested solvents and estimates uncertainty."""
    
    def __init__(self, best_models: dict, preprocessor, feature_engineer, feature_selector):
        self.best_models = best_models
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.feature_selector = feature_selector
        self.descriptor_engine = DescriptorEngine()
        
    def predict(self, solvent_names: list[str]) -> pd.DataFrame:
        """Predicts targets for a list of solvents."""
        logger.info(f"Predicting for {len(solvent_names)} solvents...")
        
        # 1. Fetch properties
        df_new = pd.DataFrame({"solvent_name": solvent_names})
        
        # Check offline DB first to speed up and avoid rate limits
        import os
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "universal_solvent_database.csv")
        
        if os.path.exists(db_path):
            db_df = pd.read_csv(db_path)
            if "Solvent_Name" in db_df.columns:
                db_df.rename(columns={"Solvent_Name": "solvent_name"}, inplace=True)
            if "solvent_name" in db_df.columns:
                for idx, row in df_new.iterrows():
                    search_val = str(row["solvent_name"]).lower()
                    match = db_df[
                        (db_df["solvent_name"].str.lower() == search_val) |
                        (db_df.get("IUPACName", pd.Series(dtype=str)).str.lower() == search_val) |
                        (db_df.get("PubChem_CommonName", pd.Series(dtype=str)).str.lower() == search_val)
                    ]
                    if not match.empty:
                        for col in match.columns:
                            if col != "solvent_name":
                                if col not in df_new.columns or pd.isna(df_new.at[idx, col]):
                                    df_new.at[idx, col] = match.iloc[0][col]
        
        # Fallback to API for any solvents or columns still missing
        df_new = self.descriptor_engine.enrich_dataset(df_new, "solvent_name")
        
        # Normalize SMILES column — the offline DB may name it 'SMILES', 'smiles', or 'IsomericSMILES'
        for smiles_variant in ["IsomericSMILES", "smiles", "SMILES"]:
            if smiles_variant in df_new.columns and "SMILES" not in df_new.columns:
                df_new.rename(columns={smiles_variant: "SMILES"}, inplace=True)
            elif smiles_variant in df_new.columns and smiles_variant != "SMILES":
                # Fill blanks in SMILES from the variant column
                df_new["SMILES"] = df_new["SMILES"].fillna(df_new[smiles_variant])
        
        if "SMILES" not in df_new.columns:
            df_new["SMILES"] = ""
        df_new["SMILES"] = df_new["SMILES"].fillna("")
        
        successful_solvents = df_new[df_new["SMILES"] != ""].copy()
        failed_solvents = df_new[df_new["SMILES"] == ""]["solvent_name"].tolist()
        
        if successful_solvents.empty:
            logger.warning("Failed to fetch properties for all requested solvents.")
            return pd.DataFrame()
            
        # 2. Preprocess
        # Apply imputation/scaling based on training data
        numeric_cols = self.preprocessor.numeric_cols
        categorical_cols = self.preprocessor.categorical_cols
        
        # Ensure new data has same columns as training data before imputation
        for col in numeric_cols:
            if col not in successful_solvents.columns:
                successful_solvents[col] = np.nan
        for col in categorical_cols:
            if col not in successful_solvents.columns:
                successful_solvents[col] = "Unknown"
                
        # Fill missing with zeros/unknown for now if missing from pubchem
        successful_solvents[numeric_cols] = successful_solvents[numeric_cols].fillna(0)
        successful_solvents[categorical_cols] = successful_solvents[categorical_cols].fillna("Unknown")
        
        X_scaled = self.preprocessor.scale_features(successful_solvents, is_train=False)
        
        # 2.5 Feature Engineering
        X_eng = self.feature_engineer.generate_features(X_scaled)
        
        # 3. Feature Selection
        # Ensure we only pass features that were selected during training
        drop_cols = [c for c in ['solvent_name', 'SMILES'] if c in X_eng.columns]
        X_for_sel = X_eng.drop(columns=drop_cols)
        X_selected = self.feature_selector.transform(X_for_sel)
        
        # 4. Predict
        results = {"solvent_name": successful_solvents["solvent_name"].tolist(), "XLogP": successful_solvents.get("XLogP", pd.Series([0]*len(successful_solvents))).tolist()}
        
        for target, model_info in self.best_models.items():
            model = model_info["model"]
            
            # Predict
            if hasattr(model, "predict"):
                if "Gaussian Process" in model_info["name"]:
                    preds, std_dev = model.predict(X_selected, return_std=True)
                    results[f"{target}_Prediction"] = preds
                    # 95% CI
                    results[f"{target}_CI_Lower"] = preds - 1.96 * std_dev
                    results[f"{target}_CI_Upper"] = preds + 1.96 * std_dev
                else:
                    preds = model.predict(X_selected)
                    results[f"{target}_Prediction"] = preds
                    
                    # Use CV RMSE as uncertainty proxy
                    cv_rmse = model_info["CV_RMSE"]
                    results[f"{target}_CI_Lower"] = preds - 1.96 * cv_rmse
                    results[f"{target}_CI_Upper"] = preds + 1.96 * cv_rmse
                    
            # Clip bounds 0-100%
            results[f"{target}_Prediction"] = np.clip(results[f"{target}_Prediction"], 0, 100)
            results[f"{target}_CI_Lower"] = np.clip(results[f"{target}_CI_Lower"], 0, 100)
            results[f"{target}_CI_Upper"] = np.clip(results[f"{target}_CI_Upper"], 0, 100)
            
            # Formatting
            results[f"{target}_Prediction"] = np.round(results[f"{target}_Prediction"], 1)
            results[f"{target}_CI_Lower"] = np.round(results[f"{target}_CI_Lower"], 1)
            results[f"{target}_CI_Upper"] = np.round(results[f"{target}_CI_Upper"], 1)

        result_df = pd.DataFrame(results)
        
        # Determine Compatibility
        # E.g., if predicted SNE and Glucose Uptake are high, it's compatible
        if "glucose_uptake" in self.best_models and "substrate_conversion" in self.best_models:
             result_df["Compatibility"] = np.where(
                 (result_df["glucose_uptake_Prediction"] > 50) & (result_df["substrate_conversion_Prediction"] > 50),
                 "Highly Compatible",
                 np.where(
                     (result_df["glucose_uptake_Prediction"] > 20) | (result_df["substrate_conversion_Prediction"] > 20),
                     "Moderately Compatible",
                     "Incompatible"
                 )
             )
        
        if failed_solvents:
            logger.warning(f"Failed to fetch: {failed_solvents}")
            
        smiles_list = successful_solvents.get("SMILES", pd.Series([np.nan]*len(successful_solvents))).tolist()
        return result_df, X_selected, smiles_list

    def fetch_descriptors(self, solvent_names: list[str]) -> pd.DataFrame:
        """Fetches descriptors for a list of solvents from offline DB and PubChem."""
        df_new = pd.DataFrame({"solvent_name": solvent_names})
        
        import os
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "universal_solvent_database.csv")
        
        if os.path.exists(db_path):
            db_df = pd.read_csv(db_path)
            if "Solvent_Name" in db_df.columns:
                db_df.rename(columns={"Solvent_Name": "solvent_name"}, inplace=True)
            if "solvent_name" in db_df.columns:
                aliases = {
                    "dmso": "dimethyl sulfoxide",
                    "butanol": "1-butanol",
                    "n-butanol": "1-butanol",
                    "thf": "tetrahydrofuran",
                    "dmf": "n,n-dimethylformamide",
                    "dcm": "dichloromethane",
                    "ipa": "2-propanol",
                    "isopropanol": "2-propanol",
                    "nmp": "n-methyl-2-pyrrolidone"
                }
                for idx, row in df_new.iterrows():
                    search_val = str(row["solvent_name"]).strip().lower()
                    if search_val in aliases:
                        search_val = aliases[search_val]
                    match = db_df[
                        (db_df["solvent_name"].str.lower() == search_val) |
                        (db_df.get("IUPACName", pd.Series(dtype=str)).str.lower() == search_val) |
                        (db_df.get("PubChem_CommonName", pd.Series(dtype=str)).str.lower() == search_val)
                    ]
                    if not match.empty:
                        for col in match.columns:
                            if col != "solvent_name":
                                if col not in df_new.columns or pd.isna(df_new.at[idx, col]):
                                    df_new.at[idx, col] = match.iloc[0][col]
        
        # Fallback to API for any solvents or columns still missing
        missing_mask = df_new["Molecular_Weight"].isna() if "Molecular_Weight" in df_new.columns else pd.Series([True]*len(df_new))
        if missing_mask.any():
            missing_df = df_new[missing_mask].copy()
            fetched_missing = self.descriptor_engine.enrich_dataset(missing_df, "solvent_name")
            df_new.update(fetched_missing)
        
        if "SMILES" not in df_new.columns:
            df_new["SMILES"] = ""
        df_new["SMILES"] = df_new["SMILES"].fillna("")
        
        # Enforce keeping only DB columns + SMILES
        if os.path.exists(db_path):
            db_cols = pd.read_csv(db_path, nrows=0).columns.tolist()
            db_cols_normalized = ["solvent_name" if c == "Solvent_Name" else c for c in db_cols]
            for c in db_cols_normalized:
                if c not in df_new.columns:
                    df_new[c] = np.nan
            keep_cols = db_cols_normalized + (["SMILES"] if "SMILES" not in db_cols_normalized else [])
            final_cols = [c for c in keep_cols if c in df_new.columns]
            df_new = df_new[final_cols]
            
        return df_new

    def predict_df(self, df_new: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        """Predicts targets for a pre-loaded/pre-edited dataframe of solvents."""
        logger.info(f"Predicting for preloaded dataframe with {len(df_new)} rows...")
        
        # Ensure SMILES column exists for structural drawings
        if "SMILES" not in df_new.columns:
            df_new["SMILES"] = ""
        df_new["SMILES"] = df_new["SMILES"].fillna("")
        
        successful_solvents = df_new.copy()
        
        # 2. Preprocess
        # Apply imputation/scaling based on training data
        numeric_cols = self.preprocessor.numeric_cols
        categorical_cols = self.preprocessor.categorical_cols
        
        # Ensure new data has same columns as training data before imputation
        for col in numeric_cols:
            if col not in successful_solvents.columns:
                successful_solvents[col] = np.nan
        for col in categorical_cols:
            if col not in successful_solvents.columns:
                successful_solvents[col] = "Unknown"
                
        # Fill missing with zeros/unknown for now if they are still NaN
        successful_solvents[numeric_cols] = successful_solvents[numeric_cols].fillna(0)
        successful_solvents[categorical_cols] = successful_solvents[categorical_cols].fillna("Unknown")
        
        X_scaled = self.preprocessor.scale_features(successful_solvents, is_train=False)
        
        # 2.5 Feature Engineering
        X_eng = self.feature_engineer.generate_features(X_scaled)
        
        # 3. Feature Selection
        id_cols = ['solvent_name', 'pubchem_cid', 'pubchem_commonname', 'iupacname', 'molecular_formula', 'smiles']
        drop_cols = [c for c in X_eng.columns if c.lower() in id_cols]
        X_for_sel = X_eng.drop(columns=drop_cols)
        X_selected = self.feature_selector.transform(X_for_sel)
        
        # 4. Predict
        results = {"solvent_name": successful_solvents["solvent_name"].tolist(), "LogP": successful_solvents.get("LogP", successful_solvents.get("XLogP", pd.Series([0]*len(successful_solvents)))).tolist()}
        
        for target, model_info in self.best_models.items():
            model = model_info["model"]
            
            # Predict
            if hasattr(model, "predict"):
                if "Gaussian Process" in model_info["name"]:
                    preds, std_dev = model.predict(X_selected, return_std=True)
                    results[f"{target}_Prediction"] = preds
                    results[f"{target}_Uncertainty"] = 1.96 * std_dev
                else:
                    preds = model.predict(X_selected)
                    results[f"{target}_Prediction"] = preds
                    cv_rmse = model_info["CV_RMSE"]
                    results[f"{target}_Uncertainty"] = [1.96 * cv_rmse] * len(preds)
                    
            # Formatting (Removed 0-100 hardcap to support unconstrained targets like cfu/ml)
            results[f"{target}_Prediction"] = np.round(results[f"{target}_Prediction"], 1)
            results[f"{target}_Prediction"] = np.round(results[f"{target}_Prediction"], 1)
            results[f"{target}_Uncertainty"] = np.round(results[f"{target}_Uncertainty"], 1)
 
        result_df = pd.DataFrame(results)
        
        # Dynamic targets are returned as-is without hardcoded strings.
        smiles_list = successful_solvents.get("SMILES", pd.Series([np.nan]*len(successful_solvents))).tolist()
        return result_df, X_selected, smiles_list
