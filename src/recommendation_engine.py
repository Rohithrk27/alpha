import pandas as pd
import numpy as np
from src.utils import setup_logger

logger = setup_logger(__name__)

class RecommendationEngine:
    """Recommends models and validation strategies based on dataset characteristics."""
    
    def analyze_dataset(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Analyzes the dataset and returns a recommendation profile."""
        n_samples, n_features = X.shape
        missing_pct = X.isnull().sum().sum() / (n_samples * n_features) if n_features > 0 else 0
        
        # Simple outlier heuristic: values > 3 std dev from mean
        num_cols = X.select_dtypes(include=['number']).columns
        if not num_cols.empty:
            z_scores = np.abs((X[num_cols] - X[num_cols].mean()) / X[num_cols].std())
            outlier_ratio = (z_scores > 3).sum().sum() / (n_samples * len(num_cols))
        else:
            outlier_ratio = 0
            
        logger.info(f"Dataset stats: {n_samples} samples, {n_features} features, "
                    f"{missing_pct:.1%} missing, {outlier_ratio:.1%} outliers.")
        
        recommendations = self._get_recommendations(n_samples)
        
        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "missing_pct": missing_pct,
            "outlier_ratio": outlier_ratio,
            "models": recommendations["models"],
            "validation_strategy": recommendations["validation_strategy"]
        }
        
    def _get_recommendations(self, n_samples: int) -> dict:
        """Returns recommended models and validation strategy based on sample size."""
        if n_samples < 20:
            return {
                "models": ["Gaussian Process"],
                "validation_strategy": "LOOCV"
            }
        elif n_samples <= 50:
            return {
                "models": ["Random Forest", "SVR", "XGBoost", "CatBoost"],
                "validation_strategy": "5-Fold CV"
            }
        elif n_samples <= 200:
            return {
                "models": ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "Gaussian Process"],
                "validation_strategy": "10-Fold CV + Test Set"
            }
        else:
            return {
                "models": ["Neural Networks", "XGBoost", "LightGBM", "Random Forest"],
                "validation_strategy": "10-Fold CV + Test Set"
            }
