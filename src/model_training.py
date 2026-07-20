import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from src.utils import setup_logger

logger = setup_logger(__name__)

class ModelTrainer:
    """Trains regression models based on recommendations."""
    
    def __init__(self):
        self.models = {}
        
    def get_model_instance(self, model_name: str):
        """Returns an uninstantiated sklearn model pipeline/estimator."""
        if model_name == "Linear Regression":
            return LinearRegression()
        elif model_name == "Polynomial Regression":
            return Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("lr", LinearRegression())
            ])
        elif model_name == "Ridge":
            return Ridge(alpha=1.0)
        elif model_name == "Lasso":
            return Lasso(alpha=0.1)
        elif model_name == "Random Forest":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == "SVR":
            return SVR(kernel='rbf', C=1.0)
        elif model_name == "Gaussian Process":
            # Force length_scale to 1.0 so the GP doesn't flatten out into the global mean
            kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds="fixed")
            return GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=0, alpha=0.1)
        elif model_name == "XGBoost":
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(random_state=42)
            except ImportError:
                logger.warning("XGBoost not installed. Falling back to Random Forest.")
                return RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == "CatBoost":
            try:
                from catboost import CatBoostRegressor
                return CatBoostRegressor(random_state=42, verbose=0)
            except ImportError:
                logger.warning("CatBoost not installed. Falling back to Random Forest.")
                return RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == "LightGBM":
            try:
                from lightgbm import LGBMRegressor
                return LGBMRegressor(random_state=42)
            except ImportError:
                logger.warning("LightGBM not installed. Falling back to Random Forest.")
                return RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            logger.warning(f"Unknown model '{model_name}'. Using Linear Regression.")
            return LinearRegression()

    def train_models(self, X: pd.DataFrame, y: pd.DataFrame, recommended_models: list[str]) -> dict:
        """Trains all recommended models for each target variable."""
        trained_models = {}
        
        for target in y.columns:
            logger.info(f"Training models for target: {target}")
            trained_models[target] = {}
            y_target = y[target].values
            
            for model_name in recommended_models:
                try:
                    model = self.get_model_instance(model_name)
                    model.fit(X, y_target)
                    trained_models[target][model_name] = model
                    logger.info(f"  Successfully trained {model_name}.")
                except Exception as e:
                    logger.error(f"  Failed to train {model_name} on {target}: {e}")
                    
        self.models = trained_models
        return trained_models
