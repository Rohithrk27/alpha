import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set up page config
st.set_page_config(
    page_title="Solvent Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium aesthetic look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global Background and Typography */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }
    .main { background-color: #121826; padding: 3rem; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); margin-top: 1rem; border: 1px solid #1f2937; }
    
    /* Headers with Gradients */
    h1 { font-weight: 800; background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; padding-bottom: 0.5rem;}
    h2, h3 { color: #e2e8f0; font-weight: 600; letter-spacing: -0.02em; }
    
    /* Subtitles and Text */
    p, li, .stMarkdown { color: #94a3b8; font-size: 1rem; line-height: 1.6; }
    
    /* Buttons */
    .stButton>button { 
        border-radius: 8px; transition: all 0.2s ease; background-color: #3b82f6; 
        color: white; border: none; font-weight: 500; padding: 0.5rem 1rem;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(59, 130, 246, 0.5); background-color: #2563eb; }
    
    /* Secondary Buttons */
    button[kind="secondary"] {
        background-color: #1e293b !important; border: 1px solid #334155 !important; color: #cbd5e1 !important;
    }
    button[kind="secondary"]:hover {
        border-color: #64748b !important; color: #f8fafc !important;
    }
    
    /* Dataframes and Expanders */
    .stDataFrame { border-radius: 8px; overflow: hidden; border: 1px solid #334155; }
    .stExpander { border: 1px solid #334155 !important; border-radius: 8px !important; background-color: #0f172a !important; }
    
    /* Tooltips / Info blocks */
    .stAlert { border-radius: 8px; border: none; }
    
    /* Responsive Design for Mobile/Tablet */
    @media (max-width: 768px) {
        .block-container { padding: 1rem !important; max-width: 100% !important; }
        .main { padding: 1.5rem; margin-top: 0.5rem; }
        h1 { font-size: 1.8rem; }
        h2 { font-size: 1.5rem; }
        h3 { font-size: 1.2rem; }
        .stButton>button { width: 100%; margin-bottom: 0.5rem; }
        .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; }
    }
</style>
""", unsafe_allow_html=True)

# Import new architecture modules
from src.data_manager import DataManager
from src.descriptor_engine import DescriptorEngine
from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from src.recommendation_engine import RecommendationEngine
from src.model_training import ModelTrainer
from src.model_validation import ModelValidator
from src.model_interpretation import ModelInterpreter
from src.prediction_engine import PredictionEngine
from src.visualization import VisualizationEngine

def _get_smiles_for_names(solvent_names: list) -> dict:
    """Loads SMILES from offline DB for a list of solvent names. Returns {name_lower: smiles}."""
    try:
        _db_path = os.path.join(os.path.dirname(__file__), "data", "universal_solvent_database.csv")
        if os.path.exists(_db_path):
            _db = pd.read_csv(_db_path, usecols=["Solvent_Name", "SMILES"])
            return dict(zip(_db["Solvent_Name"].str.strip().str.lower(), _db["SMILES"]))
    except Exception:
        pass
    return {}

def _add_structure_col(df: pd.DataFrame, smiles_map: dict) -> pd.DataFrame:
    """Adds a 'Structure' ImageColumn (PubChem URL) to a dataframe."""
    import urllib.parse
    
    img_urls = []
    any_valid = False
    
    for name in df["solvent_name"]:
        smiles = smiles_map.get(str(name).strip().lower(), "")
        url = None
        if smiles and str(smiles).strip() not in ["", "nan", "Unknown"]:
            encoded_smiles = urllib.parse.quote(str(smiles).strip())
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/PNG"
            any_valid = True
        img_urls.append(url)
        
    if any_valid:
        out = df.copy()
        out.insert(1, "Structure", img_urls)
        return out
        
    return df

def get_column_config(df_columns):
    configs = {}
    for col in df_columns:
        col_lower = col.lower()
        if col_lower in ["solvent_name", "solventname"]:
            configs[col] = st.column_config.TextColumn("Solvent Name", help="Name of the solvent", required=True)
        elif col_lower in ["molecular_weight", "molecularweight", "mw"]:
            configs[col] = st.column_config.NumberColumn("MW (g/mol)", format="%.2f", help="Molecular Weight")
        elif col_lower == "pubchem_cid":
            configs[col] = st.column_config.NumberColumn("PubChem CID", format="%d", help="PubChem Compound ID")
        elif col_lower == "pubchem_commonname":
            configs[col] = st.column_config.TextColumn("Common Name", help="Common or Synonymous Name")
        elif col_lower == "iupacname":
            configs[col] = st.column_config.TextColumn("IUPAC Name", help="IUPAC Standard Chemical Name")
        elif col_lower == "molecular_formula":
            configs[col] = st.column_config.TextColumn("Formula", help="Molecular Formula")
        elif col_lower == "smiles":
            configs[col] = st.column_config.TextColumn("SMILES", help="Simplified Molecular Input Line Entry System (SMILES) String")
        elif col_lower == "xlogp":
            configs[col] = st.column_config.NumberColumn("LogP", format="%.2f", help="Octanol-Water Partition Coefficient (XLogP)")
        elif col_lower in ["topological_polar_surface_area", "tpsa"]:
            configs[col] = st.column_config.NumberColumn("TPSA (Å²)", format="%.1f", help="Topological Polar Surface Area")
        elif col_lower in ["hydrogen_bond_donor_count", "hbonddonorcount"]:
            configs[col] = st.column_config.NumberColumn("HB Donors", format="%d", help="Hydrogen Bond Donor Count")
        elif col_lower in ["hydrogen_bond_acceptor_count", "hbondacceptorcount"]:
            configs[col] = st.column_config.NumberColumn("HB Acceptors", format="%d", help="Hydrogen Bond Acceptor Count")
        elif col_lower in ["rotatable_bond_count", "rotatablebondcount"]:
            configs[col] = st.column_config.NumberColumn("Rotatable Bonds", format="%d", help="Rotatable Bond Count")
        elif col_lower in ["heavy_atom_count", "heavyatomcount"]:
            configs[col] = st.column_config.NumberColumn("Heavy Atoms", format="%d", help="Heavy Atom Count")
        elif col_lower in ["aromatic_ring_count", "aromaticringcount"]:
            configs[col] = st.column_config.NumberColumn("Aromatic Rings", format="%d", help="Aromatic Ring Count")
        elif col_lower in ["fraction_csp3", "fractioncsp3"]:
            configs[col] = st.column_config.NumberColumn("Fraction CSP3", format="%.3f", help="Fraction of sp3 hybridized carbons")
        elif col_lower == "density":
            configs[col] = st.column_config.NumberColumn("Density (g/mL)", format="%.4f", help="Liquid Density")
        elif col_lower in ["dynamic_viscosity", "viscosity"]:
            configs[col] = st.column_config.NumberColumn("Viscosity (mPa·s)", format="%.4f", help="Dynamic Viscosity")
        elif col_lower in ["dielectric_constant", "dielectricconstant"]:
            configs[col] = st.column_config.NumberColumn("Dielectric Const", format="%.2f", help="Relative Permittivity")
        elif col_lower in ["surface_tension", "surfacetension"]:
            configs[col] = st.column_config.NumberColumn("Surf Tension (mN/m)", format="%.2f", help="Surface Tension")
        elif col_lower in ["vapor_pressure", "vaporpressure"]:
            configs[col] = st.column_config.NumberColumn("Vapor Pres (kPa)", format="%.3f", help="Vapor Pressure")
        elif col_lower in ["boiling_point", "boilingpoint"]:
            configs[col] = st.column_config.NumberColumn("Boiling Pt (°C)", format="%.1f", help="Normal Boiling Point")
        elif col_lower in ["water_solubility", "watersolubility"]:
            configs[col] = st.column_config.NumberColumn("Water Sol (mg/L)", format="%.2f", help="Solubility in Water")
        elif col_lower in ["hansen_dispersion", "hansen_dd"]:
            configs[col] = st.column_config.NumberColumn("Hansen δd", format="%.2f", help="Hansen Dispersion Parameter")
        elif col_lower in ["hansen_polar", "hansen_dp"]:
            configs[col] = st.column_config.NumberColumn("Hansen δp", format="%.2f", help="Hansen Polar Parameter")
        elif col_lower in ["hansen_hydrogen", "hansen_dh"]:
            configs[col] = st.column_config.NumberColumn("Hansen δh", format="%.2f", help="Hansen Hydrogen Bonding Parameter")
        elif col_lower in ["solvent_concentration", "solventconcentration", "concentration"]:
            configs[col] = st.column_config.NumberColumn("Concentration (%)", format="%.2f", min_value=0.0, max_value=100.0, help="Solvent Concentration in Experiment")
        elif col_lower in ["glucose_uptake", "glucoseuptake"]:
            configs[col] = st.column_config.NumberColumn("Glucose Uptake", format="%.2f", help="Glucose Uptake (%) - Primary Target")
        elif col_lower == "sne":
            configs[col] = st.column_config.NumberColumn("SNE", format="%.2f", help="SNE (%) - Target")
        elif col_lower in ["substrate_conversion", "substrateconversion"]:
            configs[col] = st.column_config.NumberColumn("Substrate Conv", format="%.2f", help="Substrate Conversion (%) - Target")
        else:
            configs[col] = st.column_config.Column(label=col.replace("_", " "))
    return configs


st.title("🧪 Solvent Toxicity Prediction Framework")
st.markdown("""
A scalable, data-driven hybrid experimental-ML framework to predict solvent toxicity for *Pichia kudriavzevii*.
""")

# Initialize session state for all modules
if 'page' not in st.session_state: st.session_state.page = "Data Upload & EDA"
if 'data_manager' not in st.session_state: st.session_state.data_manager = DataManager()
if 'descriptor_engine' not in st.session_state: st.session_state.descriptor_engine = DescriptorEngine()
if 'preprocessor' not in st.session_state: st.session_state.preprocessor = Preprocessor()
if 'feature_engineer' not in st.session_state: st.session_state.feature_engineer = FeatureEngineer()
if 'feature_selector' not in st.session_state: st.session_state.feature_selector = FeatureSelector()
if 'recommendation_engine' not in st.session_state: st.session_state.recommendation_engine = RecommendationEngine()
if 'model_trainer' not in st.session_state: st.session_state.model_trainer = ModelTrainer()
if 'model_validator' not in st.session_state: st.session_state.model_validator = ModelValidator()
if 'model_interpreter' not in st.session_state: st.session_state.model_interpreter = ModelInterpreter()
if 'visualization_engine' not in st.session_state or not hasattr(st.session_state.visualization_engine, 'plot_chemical_space'):
    st.session_state.visualization_engine = VisualizationEngine()

if 'recommendation' not in st.session_state: st.session_state.recommendation = None
if 'best_models' not in st.session_state: st.session_state.best_models = None
if 'evaluation_results' not in st.session_state: st.session_state.evaluation_results = None
if 'X_processed' not in st.session_state: st.session_state.X_processed = None
if 'y_processed' not in st.session_state: st.session_state.y_processed = None
if 'exp_metadata' not in st.session_state: st.session_state.exp_metadata = {"media": "", "temp": "", "shaking": "", "notes": ""}
if 'prediction_df' not in st.session_state: st.session_state.prediction_df = None
if 'pred_results' not in st.session_state: st.session_state.pred_results = None
if 'X_scaled_novel' not in st.session_state: st.session_state.X_scaled_novel = None
if 'smiles_map' not in st.session_state: st.session_state.smiles_map = {}
if 'training_lookup' not in st.session_state: st.session_state.training_lookup = {}

# Lazily rebuild smiles_map from prediction_df if it got wiped by a refresh
if not st.session_state.smiles_map and st.session_state.prediction_df is not None:
    _pdf = st.session_state.prediction_df
    if 'SMILES' in _pdf.columns:
        st.session_state.smiles_map = dict(zip(_pdf['solvent_name'].str.strip().str.lower(), _pdf['SMILES']))
st.sidebar.title("⚙️ Global Settings")
st.sidebar.info("Welcome to the **Solvent Toxicity Prediction Framework**.\n\nNavigate through the tabs in the main window to construct and deploy your machine learning models.")

if 'current_tab' not in st.session_state: st.session_state.current_tab = "Phase 1"

tcol1, tcol2, tcol3 = st.columns(3)
if tcol1.button("📊 Phase 1: Data Initialization", use_container_width=True, type="primary" if st.session_state.current_tab == "Phase 1" else "secondary"):
    st.session_state.current_tab = "Phase 1"
    st.rerun()
if tcol2.button("🧠 Phase 2: Model Training", use_container_width=True, type="primary" if st.session_state.current_tab == "Phase 2" else "secondary"):
    st.session_state.current_tab = "Phase 2"
    st.rerun()
if tcol3.button("🔮 Phase 3: High-Throughput Screening", use_container_width=True, type="primary" if st.session_state.current_tab == "Phase 3" else "secondary"):
    st.session_state.current_tab = "Phase 3"
    st.rerun()
st.markdown("<hr style='margin-top: -10px;'/>", unsafe_allow_html=True)

# --- PAGE 1: Data Upload & EDA ---
if st.session_state.current_tab == "Phase 1":
    st.header("Phase 1: Data Initialization & Validation")
    st.markdown("Before building toxicity models, we need a clean, structured dataset. Follow the steps below to initialize your solvent data, attach experimental conditions, and fetch molecular descriptors.")
    st.divider()
    
    st.markdown("### Step 1: Provide Base Data")
    st.info("💡 **Tip:** Ensure your data (CSV or Excel) contains a column for solvent names and numeric columns for your toxicity targets.", icon="ℹ️")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Experimental Data", type=["csv", "xlsx", "xls"])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Use Sample Data", help="Loads a small pre-configured dataset."):
            st.session_state.uploaded_name = "sample"
            sample_data = pd.DataFrame({
                "solvent_name": ["Water", "Ethanol", "Hexane", "Toluene"],
                "glucose_uptake": [95, 40, 5, 2],
                "substrate_conversion": [71, 55, 12, 3]
            })
            st.session_state.data_manager.load_data(sample_data)
            st.rerun()
        st.caption("⚠️ The experimental target values in this sample data are arbitrary and incorrect, intended for UI testing only.")
    with col3:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Enter Manually", help="Start with an empty table and type data in manually."):
            blank_data = pd.DataFrame([{"solvent_name": "", "glucose_uptake": 0.0, "substrate_conversion": 0.0}])
            st.session_state.data_manager.load_data(blank_data)
            st.session_state.uploaded_name = "manual_entry"
            st.rerun()



    if uploaded_file is not None:
        if st.session_state.data_manager.raw_data.empty or str(getattr(uploaded_file, 'name', '')) not in str(st.session_state.get('uploaded_name', '')):
            st.session_state.data_manager.load_data(uploaded_file)
            st.session_state.uploaded_name = uploaded_file.name

    if not st.session_state.data_manager.raw_data.empty:
        st.markdown("### Step 2: Data Editor & Descriptor Enrichment")
        st.write("Review your data below. You can automatically fetch structural molecular descriptors (like MW, LogP, TPSA) from our offline database or PubChem API by clicking the blue button.")
        
        with st.expander("🛠️ Interactive Data Editor (Manual Entry & Review)", expanded=True):
            st.info("💡 **Any changes made here are saved automatically.** If you fetch descriptors and some are missing (like experimental viscosity), you can type them directly into the cells.", icon="✏️")
            
            if st.button("➕ Inject All Database Columns for Manual Entry", help="Adds all 33 structural and macroscopic columns from the universal database for you to selectively fill in."):
                db_path = os.path.join(os.path.dirname(__file__), "data", "universal_solvent_database.csv")
                if os.path.exists(db_path):
                    db_cols = pd.read_csv(db_path, nrows=0).columns.tolist()
                    for c in db_cols:
                        c_normalized = "solvent_name" if c == "Solvent_Name" else c
                        if c_normalized != "solvent_name" and c_normalized not in st.session_state.data_manager.raw_data.columns:
                            st.session_state.data_manager.raw_data[c_normalized] = None
                            st.session_state.data_manager.processed_data[c_normalized] = None
                st.rerun()
            # Auto-cleanup any _x and _y columns from previous bugs to save user data
            df_to_clean = st.session_state.data_manager.raw_data
            cols_x = [c for c in df_to_clean.columns if str(c).endswith('_x')]
            for c_x in cols_x:
                base_col = c_x[:-2]
                c_y = base_col + '_y'
                if c_y in df_to_clean.columns:
                    df_to_clean[base_col] = df_to_clean[c_x].combine_first(df_to_clean[c_y])
                    df_to_clean.drop(columns=[c_x, c_y], inplace=True)
            st.session_state.data_manager.raw_data = df_to_clean
                
            col_config = get_column_config(st.session_state.data_manager.raw_data.columns)
            
            # Add 2D structures to Phase 1 experimental table
            _p1_smiles_map = _get_smiles_for_names(st.session_state.data_manager.raw_data["solvent_name"].tolist())
            _p1_display_df = _add_structure_col(st.session_state.data_manager.raw_data, _p1_smiles_map)
            if "Structure" in _p1_display_df.columns:
                col_config["Structure"] = st.column_config.ImageColumn("Structure", width="small")
            
            _edited_p1 = st.data_editor(
                _p1_display_df,
                use_container_width=True, 
                num_rows="dynamic",
                column_config=col_config
            )
            # Store clean data (without display-only Structure column) for training
            st.session_state.data_manager.processed_data = _edited_p1.drop(columns=["Structure"], errors="ignore")
            
            st.markdown("##### Quick Add Column")
            ccol1, ccol2 = st.columns([3, 1])
            with ccol1:
                new_col_name = st.text_input("Column Name", key="new_col_name_input", placeholder="e.g. Test_Condition", label_visibility="collapsed")
            with ccol2:
                if st.button("Add Column", use_container_width=True, type="secondary"):
                    if new_col_name and new_col_name not in st.session_state.data_manager.raw_data.columns:
                        st.session_state.data_manager.raw_data[new_col_name] = 0.0
                        st.session_state.data_manager.processed_data[new_col_name] = 0.0
                        st.rerun()
            
            # Check for missing data
            df_current = st.session_state.data_manager.processed_data
            if not df_current.empty:
                missing_counts = df_current.isnull().sum()
                missing_cols = missing_counts[missing_counts > 0]
                if not missing_cols.empty:
                    st.warning("⚠️ Some descriptors or targets currently have missing values (NaN). Please provide them manually or click 'Fetch Descriptors' below.")
                    st.write(missing_cols.to_frame("Missing Count"))
            
        st.markdown("### Step 4: Process & Fetch")
        colA, colB = st.columns(2)
        with colA:
            st.caption("✅ **Note:** The molecular descriptors fetched here are scientifically valid, sourced from real chemical databases, unlike the dummy targets.")
            if st.button("Fetch Molecular Descriptors 🧬", help="Pulls structural descriptors (MW, LogP, etc.) from the offline DB or PubChem."):
                with st.spinner("Fetching descriptors..."):
                    df = st.session_state.data_manager.processed_data
                    db_path = os.path.join(os.path.dirname(__file__), "data", "universal_solvent_database.csv")
                    enriched_df = df.copy()
                    
                    # Check offline DB first to speed up
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
                            for idx, row in enriched_df.iterrows():
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
                                            if col not in enriched_df.columns or pd.isna(enriched_df.at[idx, col]):
                                                enriched_df.at[idx, col] = match.iloc[0][col]
                                                
                    # Fallback to API for anything still missing (like Molecular_Weight)
                    missing_mask = enriched_df["Molecular_Weight"].isna() if "Molecular_Weight" in enriched_df.columns else pd.Series([True]*len(enriched_df))
                    if missing_mask.any():
                        missing_df = enriched_df[missing_mask].copy()
                        fetched_missing = st.session_state.descriptor_engine.enrich_dataset(missing_df, "solvent_name")
                        enriched_df.update(fetched_missing)
                    # Force all known universal schema columns to exist so the user can see them and manually fill them
                    if os.path.exists(db_path):
                        db_cols = pd.read_csv(db_path, nrows=0).columns.tolist()
                        for c in db_cols:
                            c_normalized = "solvent_name" if c == "Solvent_Name" else c
                            if c_normalized not in enriched_df.columns:
                                enriched_df[c_normalized] = None
                                
                        # Keep only the columns from the database schema plus original user columns
                        db_cols_normalized = ["solvent_name" if c == "Solvent_Name" else c for c in db_cols]
                        keep_cols = list(df.columns) + [c for c in db_cols_normalized if c not in df.columns]
                        final_cols = [c for c in keep_cols if c in enriched_df.columns]
                        enriched_df = enriched_df[final_cols]
                                
                    st.session_state.data_manager.processed_data = enriched_df
                    st.session_state.data_manager.raw_data = enriched_df
                    st.success("Successfully fetched chemical descriptors!")
                    st.rerun()
                
        # Show all descriptors explicitly if they have been fetched
        if any(c in st.session_state.data_manager.raw_data.columns for c in ["Molecular_Weight", "XLogP", "molecular_weight"]):
            with st.expander("View Full Fetched Descriptors Summary", expanded=False):
                st.write("Here are all the descriptors currently loaded for your solvents:")
                cols_to_show = st.session_state.data_manager.raw_data.columns.tolist()
                if "solvent_name" in cols_to_show:
                    cols_to_show.remove("solvent_name")
                    cols_to_show = ["solvent_name"] + cols_to_show
                col_config_summary = get_column_config(cols_to_show)
                st.dataframe(st.session_state.data_manager.raw_data[cols_to_show], use_container_width=True, column_config=col_config_summary)
                
        with colB:
            st.markdown("#### Finalize Data")
            st.write("Click below to finalize your data and proceed to modeling.")
            
            if st.button("Process & Analyze Data 🚀", type="primary", help="Cleans data, selects features, and prepares for machine learning."):
                with st.spinner("Processing..."):
                    df = st.session_state.data_manager.processed_data.copy()
                    
                    # Identify targets (exclude all known chemical descriptors)
                    known_desc = [
                        'molecular_weight', 'xlogp', 'logp', 'topological_polar_surface_area',
                        'hydrogen_bond_donor_count', 'hydrogen_bond_acceptor_count', 
                        'rotatable_bond_count', 'heavy_atom_count', 'aromatic_ring_count',
                        'fraction_csp3', 'density', 'dynamic_viscosity', 'dielectric_constant',
                        'dipole_moment', 'surface_tension', 'vapor_pressure', 'boiling_point',
                        'melting_point', 'flash_point', 'refractive_index', 'water_solubility',
                        'hansen_dispersion', 'hansen_polar', 'hansen_hydrogen',
                        'hansen_dd', 'hansen_dp', 'hansen_dh', 'hildebrand_parameter',
                        'critical_temperature', 'critical_pressure', 'enthalpy_of_vaporization',
                        'viscosity', 'pubchem_cid', 'solvent_concentration'
                    ]
                    num_cols = df.select_dtypes(include=['number']).columns.tolist()
                    targets = [c for c in num_cols if c.lower() not in known_desc]
                    
                    if not targets:
                        st.error("No target variables found! Ensure you have numeric columns for targets like 'glucose_uptake'.")
                    else:
                        X_raw, y = st.session_state.data_manager.get_solvents_and_targets("solvent_name", targets)
                        
                        # Merge X_raw with descriptors that are already in processed_data
                        desc_cols = [c for c in df.columns if c not in targets and c != "control"]
                        X = df.loc[X_raw.index, desc_cols].copy()
                        
                        # Preprocess
                        X_clean = st.session_state.preprocessor.clean_data(X)
                        X_scaled = st.session_state.preprocessor.scale_features(X_clean, is_train=True)
                        
                        # Engineer & Select
                        X_eng = st.session_state.feature_engineer.generate_features(X_scaled)
                        
                        # Drop non-predictive identification columns before feature selection
                        id_cols = ['solvent_name', 'pubchem_cid', 'pubchem_commonname', 'iupacname', 'molecular_formula', 'smiles']
                        drop_cols = [c for c in X_eng.columns if c.lower() in id_cols]
                        X_for_sel = X_eng.drop(columns=drop_cols)
                        
                        X_selected = st.session_state.feature_selector.fit_transform(X_for_sel)
                        
                        st.session_state.X_processed = X_selected
                        st.session_state.y_processed = y
                        
                        # Clear old trained models to prevent state desync
                        st.session_state.evaluation_results = None
                        st.session_state.best_models = None
                        
                        # Recommendation Engine
                        st.session_state.recommendation = st.session_state.recommendation_engine.analyze_dataset(X_selected, y)
                        
            if st.session_state.get("recommendation") is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                st.success("✅ Data is processed and ready for modeling!")
                if st.button("Proceed to Phase 2 ➔", type="primary"):
                    st.session_state.current_tab = "Phase 2"
                    st.rerun()

# --- PAGE 2: Model Training & Eval ---
if st.session_state.current_tab == "Phase 2":
    st.header("Phase 2: Model Training & Evaluation")
    
    if st.session_state.recommendation is None:
        st.warning("Please upload and process your dataset in Phase 1 first.")
    else:
        rec = st.session_state.recommendation
        

        st.markdown("### Step 1: Dataset Profile & AI Recommendations")
        st.markdown("Based on the shape and quality of your processed data, the Recommendation Engine suggests the following strategy:")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Samples", rec["n_samples"])
        col2.metric("Features (After Selection)", rec["n_features"])
        col3.metric("Missing Data", f"{rec['missing_pct']:.1%}", help="Percentage of missing data that was imputed.")
        col4.metric("Outliers", f"{rec['outlier_ratio']:.1%}", help="Percentage of statistical outliers detected and handled.")
        
        st.success(f"**Recommended Validation Strategy:** {rec['validation_strategy']}\n\n**Recommended Algorithms:** {', '.join(rec['models'])}")
        
        st.markdown("### Step 2: Train & Validate Models")
        st.write("Click below to train all recommended algorithms and cross-validate them to find the absolute best predictor for your targets.")
        
        if st.button("Train Recommended Models 🚀", type="primary"):
            with st.spinner("Training models in parallel..."):
                X = st.session_state.X_processed
                y = st.session_state.y_processed
                
                # Train
                trained_models = st.session_state.model_trainer.train_models(X, y, rec["models"])
                
                # Validate
                eval_results = st.session_state.model_validator.evaluate(X, y, trained_models, rec["validation_strategy"])
                st.session_state.evaluation_results = eval_results
                
                # Select best
                best_models = st.session_state.model_validator.select_best_models(eval_results)
                st.session_state.best_models = best_models
                
                # Build exact lookup table for training solvents → guarantees exact match in predictions
                training_lookup = {}
                proc_data = st.session_state.data_manager.processed_data
                y_data = st.session_state.y_processed
                if "solvent_name" in proc_data.columns:
                    for i, row in proc_data.iterrows():
                        name_key = str(row["solvent_name"]).strip().lower()
                        training_lookup[name_key] = {}
                        for target_col in y_data.columns:
                            if i in y_data.index and target_col in y_data.columns:
                                training_lookup[name_key][target_col] = y_data.at[i, target_col]
                st.session_state.training_lookup = training_lookup
                
        if st.session_state.evaluation_results is not None:
            st.divider()
            st.markdown("### Step 3: Performance Analysis")
            
            for target in st.session_state.y_processed.columns:
                st.markdown(f"#### Target: {str(target).replace('_', ' ')}")
                fig = st.session_state.visualization_engine.plot_model_comparison(st.session_state.evaluation_results, target)
                if fig: st.pyplot(fig)
                
            st.markdown("#### Actual vs Predicted (Best Models)")
            st.info("A perfect model would have all points falling exactly on the diagonal dashed line.")
            fig2 = st.session_state.visualization_engine.plot_actual_vs_predicted(st.session_state.best_models, st.session_state.y_processed)
            if fig2: st.pyplot(fig2)
            
            st.divider()
            st.markdown("### Step 4: Explainable AI (SHAP)")
            st.write("Understand *why* the model makes its predictions. Which chemical descriptors are driving toxicity?")
            
            target_to_explain = st.selectbox("Select target to explain:", st.session_state.y_processed.columns, format_func=lambda x: str(x).replace("_", " "))
            if st.button("Generate SHAP Summary", help="Calculates SHAP (SHapley Additive exPlanations) values."):
                with st.spinner("Calculating SHAP values (this may take a moment)..."):
                    best_mod_info = st.session_state.best_models[target_to_explain]
                    fig_shap = st.session_state.model_interpreter.generate_shap_summary(best_mod_info["model"], st.session_state.X_processed)
                    if fig_shap:
                        st.pyplot(fig_shap)
                        with st.expander("📖 How to Interpret this SHAP Plot", expanded=True):
                            st.markdown(
                                """
                                **SHAP (SHapley Additive exPlanations)** breaks down exactly how the AI makes its decisions. Here is how to read the chart above:
                                
                                * ↕️ **Y-Axis (Top to Bottom):** The most important features are at the top. The model relies heavily on these to predict your target.
                                * ↔️ **X-Axis (Left to Right):** Dots on the **right side** of the center line mean that feature pushed the prediction *higher* (e.g., higher toxicity). Dots on the **left side** mean it pushed the prediction *lower*.
                                * 🎨 **Color (Red vs Blue):** **Red dots** mean the solvent had a *High* value for that specific feature. **Blue dots** mean it had a *Low* value. 
                                
                                **Example:** If you see a cluster of **Red dots** on the **Far Right** for *LogP*, it means: "When a solvent has a HIGH LogP, it drastically INCREASES the target prediction."
                                """
                            )
                    else:
                        st.warning("SHAP explanation not available for this model type or SHAP is not installed.")
            
            st.divider()
            if st.button("Proceed to Phase 3: High-Throughput Screening ➔", type="primary"):
                st.session_state.current_tab = "Phase 3"
                st.rerun()

# --- PAGE 3: Prediction & Inference ---
if st.session_state.current_tab == "Phase 3":
    st.header("Phase 3: High-Throughput Screening")
    
    if st.session_state.best_models is None:
        st.warning("Please train and select your models in Phase 2 before attempting predictions.")
    else:
        st.markdown("### Predict New Solvents")
        st.write("Using the highest-performing machine learning model found during Phase 2 cross-validation, we can now predict the toxicity/targets of completely novel, untested solvents.")
        
        st.info("💡 **How it works:** \n1. Enter the names of the solvents below.\n2. Click 'Initialize Solvents & Fetch Descriptors' to compile their properties.\n3. Review the properties in the slideable table and fill in any missing values manually.\n4. Click 'Run Model Prediction' to compute targets.", icon="ℹ️")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            solvents_input = st.text_area("Enter solvent names (comma-separated):", "Cyclohexanone, Butanol, Pyridine", help="You can enter common names or IUPAC names. The descriptor engine will automatically parse them.")
            btn_fetch = st.button("1. Initialize Solvents & Fetch Descriptors 🧬", type="secondary", use_container_width=True)
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True) # padding
            btn_scan_all = st.button("🌟 Scan Entire Database", type="primary", use_container_width=True, help="Automatically evaluate all solvents in the internal database to find the absolute best match.")
            
        solvent_list = None
        if btn_fetch:
            solvent_list = [s.strip() for s in solvents_input.split(",") if s.strip()]
        elif btn_scan_all:
            import os
            import pandas as pd
            db_path = os.path.join(os.path.dirname(__file__), "data", "universal_solvent_database.csv")
            if os.path.exists(db_path):
                db_df = pd.read_csv(db_path, usecols=["Solvent_Name"])
                db_solvents = db_df["Solvent_Name"].dropna().unique().tolist()
                solvent_list = [str(s).strip() for s in db_solvents if str(s).strip()]
            else:
                st.error("Offline database not found.")
            
        if solvent_list:
            with st.spinner("Fetching descriptors from offline database and API..."):
                pred_engine = PredictionEngine(
                    st.session_state.best_models,
                    st.session_state.preprocessor,
                    st.session_state.feature_engineer,
                    st.session_state.feature_selector
                )
                prediction_df = pred_engine.fetch_descriptors(solvent_list)
                st.session_state.prediction_df = prediction_df
                # Save the SMILES map right now before the user can edit them away
                if "SMILES" in prediction_df.columns:
                    st.session_state.smiles_map = dict(zip(prediction_df["solvent_name"].str.strip().str.lower(), prediction_df["SMILES"]))
                else:
                    st.session_state.smiles_map = {}
                st.session_state.pred_results = None # Reset previous prediction results
                st.session_state.X_scaled_novel = None
                st.rerun()
        elif btn_fetch:
            st.error("Please enter at least one solvent.")
                
        if st.session_state.prediction_df is not None:
            st.divider()
            st.markdown("### ✏️ Step 2: Edit & Review Descriptors")
            st.write("Slide/scroll the table below to view all factors. Double-click any cell to manually edit or insert missing values.")
            
            pred_col_config = get_column_config(st.session_state.prediction_df.columns)
            
            # Add 2D structures to Phase 3 descriptor review table
            _p3_smiles_map = st.session_state.get("smiles_map", {}) or _get_smiles_for_names(st.session_state.prediction_df["solvent_name"].tolist())
            _p3_display_df = _add_structure_col(st.session_state.prediction_df, _p3_smiles_map)
            if "Structure" in _p3_display_df.columns:
                pred_col_config["Structure"] = st.column_config.ImageColumn("Structure", width="small")
            
            # Show interactive table for editing
            edited_pred_df = st.data_editor(
                _p3_display_df,
                use_container_width=True,
                num_rows="dynamic",
                column_config=pred_col_config,
                key="prediction_data_editor_widget"
            )
            
            # Add a button to predict on this edited table
            if st.button("2. Run Model Prediction 🚀", type="primary", use_container_width=True):
                with st.spinner("Running predictions through the model..."):
                    pred_engine = PredictionEngine(
                        st.session_state.best_models,
                        st.session_state.preprocessor,
                        st.session_state.feature_engineer,
                        st.session_state.feature_selector
                    )
                    
                    # Strip any display-only columns that were injected for the table view
                    clean_pred_df = edited_pred_df.drop(columns=["Structure", "_SMILES"], errors="ignore")
                    results_df, X_scaled_novel, smiles_list = pred_engine.predict_df(clean_pred_df)
                    
                    # Rebuild SMILES from saved map (survives data editor edits)
                    smiles_map = st.session_state.get("smiles_map", {}) or _get_smiles_for_names(results_df["solvent_name"].tolist())
                    st.session_state.smiles_map = smiles_map  # ensure it's cached
                    
                    if not results_df.empty:
                        # Insert 2D structure images via the shared helper
                        idx_insert = 1
                        results_df = _add_structure_col(results_df, smiles_map)
                        if "Structure" in results_df.columns:
                            # Also save SMILES for PDF report generation
                            results_df["_SMILES"] = [smiles_map.get(str(n).strip().lower(), "") for n in results_df["solvent_name"]]
                            idx_insert = 2
                        
                        # ── EXACT MATCH OVERRIDE FOR TRAINING SOLVENTS ──────────────
                        # If a solvent was in the training set, use its exact experimental
                        # value instead of the model's approximation.
                        training_lookup = st.session_state.get("training_lookup", {})
                        if training_lookup:
                            targets = list(st.session_state.best_models.keys())
                            for i, row in results_df.iterrows():
                                name_key = str(row["solvent_name"]).strip().lower()
                                if name_key in training_lookup:
                                    for target in targets:
                                        pred_col = f"{target}_Prediction"
                                        unc_col = f"{target}_Uncertainty"
                                        if pred_col in results_df.columns and target in training_lookup[name_key]:
                                            results_df.at[i, pred_col] = round(training_lookup[name_key][target], 1)
                                        if unc_col in results_df.columns:
                                            results_df.at[i, unc_col] = 0.0
                        
                        # Calculate Overall Compatibility Score using Global Min-Max & Geometric Mean
                        target_preds = [f"{t}_Prediction" for t in st.session_state.best_models.keys() if f"{t}_Prediction" in results_df.columns]
                        if target_preds:
                            import numpy as np
                            norm_df = results_df[target_preds].copy()
                            y_train = st.session_state.get("y_processed", pd.DataFrame())
                            
                            for col in norm_df.columns:
                                target_name = col.replace("_Prediction", "")
                                # Use global bounds from training data if available
                                if target_name in y_train.columns:
                                    c_min, c_max = y_train[target_name].min(), y_train[target_name].max()
                                else:
                                    c_min, c_max = norm_df[col].min(), norm_df[col].max()
                                    
                                if c_max > c_min:
                                    norm_val = (norm_df[col] - c_min) / (c_max - c_min)
                                    # Clip to [0,1] just in case a novel solvent goes slightly outside bounds
                                    norm_df[col] = np.clip(norm_val, 0.001, 1.0)
                                else:
                                    norm_df[col] = 1.0
                            
                            # Geometric mean: requires ALL targets to be high. If one is low, the whole score tanks.
                            geom_mean = np.exp(np.log(norm_df).mean(axis=1))
                            scores = [f"{round(s * 100, 1)}%" for s in geom_mean]
                        else:
                            scores = ["0%"] * len(results_df)
                            
                        results_df.insert(idx_insert, "Overall_Compatibility", scores)
                        idx_insert += 1
                        
                        # Sort by highest overall compatibility
                        results_df["_sort_val"] = results_df["Overall_Compatibility"].str.replace("%", "").astype(float)
                        results_df.sort_values(by="_sort_val", ascending=False, inplace=True)
                        results_df.drop(columns=["_sort_val"], inplace=True)
                        
                        # Reset index and shift by 1 for clean 1-based ranking display
                        results_df.reset_index(drop=True, inplace=True)
                        results_df.index = results_df.index + 1
                        
                        # Set up UI columns
                        col_config_pred = {}
                        for c in results_df.columns:
                            if "_Prediction" in c:
                                t_name = c.replace("_Prediction", "")
                                col_config_pred[c] = st.column_config.NumberColumn(f"{t_name} (Predicted)", format="%.1f")
                                # Try to grab Uncertainty column
                                if f"{t_name}_Uncertainty" in results_df.columns:
                                    col_config_pred[f"{t_name}_Uncertainty"] = st.column_config.NumberColumn(f"± Uncertainty (95% CI)", format="%.1f")
                                
                        st.session_state.pred_results = results_df
                        st.session_state.X_scaled_novel = X_scaled_novel
                        st.rerun()
                    else:
                        st.error("Prediction failed. Please ensure the inputs in the table are valid.")
                        
        if st.session_state.pred_results is not None:
            results_df = st.session_state.pred_results
            X_scaled_novel = st.session_state.X_scaled_novel
            
            st.divider()
            st.markdown("### 🏆 Prediction Results")
            st.info("💡 **What do the metrics mean?**\n- **Overall_Compatibility**: Measures how close the solvent is to an ideal 100% reference control.\n- **CI_Lower / CI_Upper**: The 95% Confidence Interval. This is the model's 'margin of error'. We are 95% confident the true value falls between these bounds.")
            
            # Highlight the best solvent
            best_solvent = results_df.iloc[0]['solvent_name']
            best_score = results_df.iloc[0]['Overall_Compatibility']
            st.success(f"🏆 **Top Recommended Solvent:** {best_solvent} (Compatibility: {best_score})")

            # Make UI readable by replacing underscores and registering ImageColumn
            display_results_df = results_df.drop(columns=["_SMILES"], errors="ignore")
            col_config_results = {col: st.column_config.Column(label=col.replace("_", " ")) for col in display_results_df.columns}
            if "Structure" in display_results_df.columns:
                col_config_results["Structure"] = st.column_config.ImageColumn("Structure", width="small")
                
            st.dataframe(display_results_df, use_container_width=True, column_config=col_config_results)
            
            # Generate PCA plot
            st.divider()
            st.markdown("### 🌌 Chemical Space (PCA)")
            st.write("Visualizing your novel predictions against your original training dataset.")
            fig_pca = st.session_state.visualization_engine.plot_chemical_space(
                st.session_state.X_processed, X_scaled_novel, 
                st.session_state.data_manager.processed_data["solvent_name"].tolist(),
                results_df["solvent_name"].tolist()
            )
            if fig_pca:
                st.pyplot(fig_pca)
            
            # Generate Bar Charts
            st.divider()
            st.markdown("### 📊 Target Predictions (Top 20 Solvents)")
            st.write("Visualizing the highest recommended solvents and their 95% Confidence Interval error bars.")
            fig_bar = st.session_state.visualization_engine.plot_prediction_bar_charts(results_df)
            if fig_bar:
                st.pyplot(fig_bar)
            
            st.divider()
            colD, colE = st.columns(2)
            with colD:
                # Add a download button for the predictions CSV
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV Table",
                    data=csv,
                    file_name='solvent_toxicity_predictions.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            with colE:
                if st.button("📄 Generate Comprehensive PDF Lab Report", type="primary", use_container_width=True):
                    with st.spinner("Compiling PDF report (generating images)..."):
                        from src.report_generator import ReportGenerator
                        import tempfile
                        
                        img_paths = {}
                        temp_dir = tempfile.mkdtemp()
                        
                        if fig_pca:
                            pca_path = os.path.join(temp_dir, "pca.png")
                            fig_pca.savefig(pca_path, bbox_inches='tight')
                            img_paths['pca'] = pca_path
                            
                        fig_actual = st.session_state.visualization_engine.plot_actual_vs_predicted(st.session_state.best_models, st.session_state.y_processed)
                        if fig_actual:
                            act_path = os.path.join(temp_dir, "actual.png")
                            fig_actual.savefig(act_path, bbox_inches='tight')
                            img_paths['actual_vs_predicted'] = act_path
                            
                        target_to_explain = list(st.session_state.y_processed.columns)[0]
                        fig_shap = st.session_state.model_interpreter.generate_shap_summary(st.session_state.best_models[target_to_explain]["model"], st.session_state.X_processed)
                        if fig_shap:
                            shap_path = os.path.join(temp_dir, "shap.png")
                            fig_shap.savefig(shap_path, bbox_inches='tight', dpi=150)
                            img_paths['shap'] = shap_path
                            
                        # Generate Combined Bar Chart
                        fig_combined = st.session_state.visualization_engine.plot_combined_bar_charts(results_df, st.session_state.data_manager.processed_data)
                        if fig_combined:
                            combined_path = os.path.join(temp_dir, "combined_bar.png")
                            fig_combined.savefig(combined_path, bbox_inches='tight')
                            img_paths['combined_bar'] = combined_path
                            
                        # Generate Structure Images for Top 3
                        struct_paths = []
                        if "_SMILES" in results_df.columns:
                            try:
                                from rdkit import Chem
                                from rdkit.Chem import Draw
                                for i, s in enumerate(results_df["_SMILES"].head(3)):
                                    if pd.notna(s) and str(s).strip() not in ["Unknown", "nan", ""]:
                                        mol = Chem.MolFromSmiles(str(s).strip())
                                        if mol:
                                            img_path = os.path.join(temp_dir, f"struct_{i}.png")
                                            Draw.MolToFile(mol, img_path, size=(250, 250))
                                            struct_paths.append(img_path)
                            except ImportError:
                                pass
                        img_paths['structures'] = struct_paths
                        
                        generator = ReportGenerator(output_dir=temp_dir)
                        pdf_path = generator.generate_report(st.session_state.evaluation_results, st.session_state.best_models, results_df, img_paths)
                        
                        with open(pdf_path, "rb") as pdf_file:
                            PDFbyte = pdf_file.read()
                            
                        st.download_button(label="⬇️ Download PDF Report", 
                                           data=PDFbyte,
                                           file_name="Solvent_Lab_Report.pdf",
                                           mime='application/octet-stream',
                                           use_container_width=True)
