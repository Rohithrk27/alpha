import pandas as pd
import matplotlib.pyplot as plt
from src.utils import setup_logger

logger = setup_logger(__name__)

class ModelInterpreter:
    """Uses SHAP for Explainable AI (if available)."""
    
    def __init__(self):
        try:
            import shap
            self.shap = shap
            self.has_shap = True
        except ImportError:
            self.has_shap = False
            logger.warning("SHAP not installed. Explainable AI features will be disabled.")
            
    def generate_shap_summary(self, model, X: pd.DataFrame) -> plt.Figure:
        """Generates a SHAP summary plot for a given model."""
        if not self.has_shap:
            return None
            
        try:
            # Tree explainer is typically faster for Random Forest / XGBoost
            if hasattr(model, "estimators_") or "XGB" in str(type(model)):
                explainer = self.shap.TreeExplainer(model)
            else:
                # Fallback to KernelExplainer for others (can be slow)
                # Use a background summary to speed up
                background = self.shap.kmeans(X, 10)
                explainer = self.shap.KernelExplainer(model.predict, background)
                
            shap_values = explainer.shap_values(X)
            
            fig = plt.figure(figsize=(10, 6))
            self.shap.summary_plot(shap_values, X, show=False)
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Failed to generate SHAP summary: {e}")
            return None
