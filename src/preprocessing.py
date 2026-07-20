import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils import setup_logger

logger = setup_logger(__name__)

class Preprocessor:
    """Handles missing value imputation, duplicate removal, and scaling."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_cols = []
        self.categorical_cols = []
        self.fitted_numeric_cols = []
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicates and handles missing values."""
        df_clean = df.drop_duplicates().copy()
        
        self.numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df_clean.select_dtypes(exclude=['number']).columns.tolist()
        
        # Simple imputation: median for numeric, mode for categorical
        if self.numeric_cols:
            df_clean[self.numeric_cols] = df_clean[self.numeric_cols].fillna(df_clean[self.numeric_cols].median())
        
        if self.categorical_cols:
            for col in self.categorical_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown")
                
        logger.info("Data cleaned and missing values imputed.")
        return df_clean

    def scale_features(self, X: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Scales numeric features using StandardScaler."""
        X_scaled = X.copy()
        
        if is_train:
            self.fitted_numeric_cols = X_scaled.select_dtypes(include=['number']).columns.tolist()
            
        if not self.fitted_numeric_cols:
            return X_scaled
            
        # Ensure fitted columns exist in inference data
        for col in self.fitted_numeric_cols:
            if col not in X_scaled.columns:
                X_scaled[col] = 0.0
                
        if is_train:
            X_scaled[self.fitted_numeric_cols] = self.scaler.fit_transform(X_scaled[self.fitted_numeric_cols])
            logger.info("Fitted and transformed numeric features.")
        else:
            X_scaled[self.fitted_numeric_cols] = self.scaler.transform(X_scaled[self.fitted_numeric_cols])
            logger.info("Transformed numeric features using fitted scaler.")
            
        return X_scaled
