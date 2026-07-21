import pandas as pd
import numpy as np
from src.utils import setup_logger

logger = setup_logger(__name__)

class FeatureSelector:
    """Removes constant and highly correlated features."""
    
    def __init__(self, correlation_threshold: float = 0.9):
        self.correlation_threshold = correlation_threshold
        self.selected_features = []
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature selection...")
        X_new = X.copy()
        
        # Remove constant features
        constant_cols = [col for col in X_new.columns if X_new[col].nunique() <= 1]
        X_new.drop(columns=constant_cols, inplace=True)
        logger.info(f"Dropped {len(constant_cols)} constant features.")
        
        # Adjust correlation threshold dynamically based on sample size
        n_samples = len(X_new)
        dynamic_threshold = self.correlation_threshold
        if n_samples < 20:
            logger.info(f"Small dataset ({n_samples} rows) detected. Aggressively selecting only essential descriptors.")
            # Explicitly lock onto the top ranked biological descriptors for small datasets to prevent overfitting
            # Ranked strictly by biological importance (LogP > Polarity > Solubility > Viscosity > HSP > Surface Tension > MW)
            ranked_essentials = [
                'xlogp', 'logp',                            # Rank 1: Membrane partitioning
                'dielectric_constant',                      # Rank 2: Polarity
                'water_solubility',                         # Rank 3: Aqueous partitioning
                'dynamic_viscosity', 'viscosity',           # Rank 4: Mass transfer
                'hansen_dispersion', 'hansen_polar', 'hansen_hydrogen', 'hildebrand_parameter', # Rank 5: HSP
                'surface_tension',                          # Rank 6: Wetting
                'molecular_weight'                          # Rank 7: Size
            ]
            
            cols_to_keep = []
            for candidate in ranked_essentials:
                # Find matching columns in X_new (case insensitive)
                matches = [c for c in X_new.columns if c.lower() == candidate and c not in cols_to_keep]
                cols_to_keep.extend(matches)
                if len(cols_to_keep) >= 4:
                    break
            
            # Strictly cap at 4 features to prevent Curse of Dimensionality on small datasets
            cols_to_keep = cols_to_keep[:4]
            
            # If we somehow found zero essentials, fallback to the first 2 numeric columns
            if not cols_to_keep:
                numeric_cols = X_new.select_dtypes(include=['number']).columns.tolist()
                cols_to_keep = numeric_cols[:2]
                
            cols_to_drop = [c for c in X_new.columns if c not in cols_to_keep]
            X_new.drop(columns=cols_to_drop, inplace=True)
        else:
            # Remove highly correlated features (only for numeric columns)
            numeric_cols = X_new.select_dtypes(include=['number']).columns
            corr_matrix = X_new[numeric_cols].corr().abs()
            
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > dynamic_threshold)]
            
            X_new.drop(columns=to_drop, inplace=True)
            logger.info(f"Dropped {len(to_drop)} highly correlated features (threshold={self.correlation_threshold}).")
        
        self.selected_features = X_new.columns.tolist()
        return X_new

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features:
            raise ValueError("Selector has not been fitted yet.")
        
        # Only keep selected features that are actually in X
        cols_to_keep = [col for col in self.selected_features if col in X.columns]
        return X[cols_to_keep].copy()
