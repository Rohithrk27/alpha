import pandas as pd
from typing import Dict, Any, Tuple
from src.utils import setup_logger

logger = setup_logger(__name__)

class DataManager:
    """Handles loading and managing the experimental dataset and metadata."""
    
    def __init__(self):
        self.metadata: Dict[str, Any] = {}
        self.raw_data: pd.DataFrame = pd.DataFrame()
        self.processed_data: pd.DataFrame = pd.DataFrame()

    def load_data(self, file_or_df) -> pd.DataFrame:
        """Loads data from a CSV file path, file object, or pandas DataFrame."""
        try:
            if isinstance(file_or_df, pd.DataFrame):
                self.raw_data = file_or_df.copy()
            else:
                is_excel = False
                if hasattr(file_or_df, "name") and file_or_df.name.endswith((".xlsx", ".xls")):
                    is_excel = True
                elif isinstance(file_or_df, str) and file_or_df.endswith((".xlsx", ".xls")):
                    is_excel = True
                    
                if is_excel:
                    self.raw_data = pd.read_excel(file_or_df)
                else:
                    self.raw_data = pd.read_csv(file_or_df)
            logger.info(f"Successfully loaded data with {len(self.raw_data)} rows.")
            self.processed_data = self.raw_data.copy()
            return self.raw_data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def set_metadata(self, organism: str = "Pichia kudriavzevii", **kwargs):
        """Stores experimental metadata for reproducibility."""
        self.metadata = {
            "Organism": organism,
            **kwargs
        }
        logger.info(f"Metadata updated: {self.metadata}")

    def get_solvents_and_targets(self, solvent_col: str, target_cols: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separates the solvent names (features) from the targets."""
        if self.processed_data.empty:
            raise ValueError("No data available. Load data first.")
        
        # Filter out control rows for modeling
        df = self.processed_data[~self.processed_data[solvent_col].astype(str).str.lower().str.contains("control", na=False)]
        
        X_info = df[[solvent_col]].copy()
        y = df[target_cols].copy()
        
        return X_info, y
