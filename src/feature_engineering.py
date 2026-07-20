import pandas as pd
import numpy as np
from src.utils import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    """Creates combined/polynomial features."""
    
    def generate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        
        # Example polynomial features
        if 'XLogP' in X_new.columns:
            X_new['XLogP_squared'] = X_new['XLogP'] ** 2
            
        if 'XLogP' in X_new.columns and 'Molecular_Weight' in X_new.columns:
            X_new['XLogP_x_MW'] = X_new['XLogP'] * X_new['Molecular_Weight']
            
        logger.info(f"Engineered {len(X_new.columns) - len(X.columns)} new features.")
        return X_new
