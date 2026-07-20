import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.utils import setup_logger

logger = setup_logger(__name__)

class ModelValidator:
    """Evaluates trained models using the recommended validation strategy."""
    
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame, trained_models: dict, validation_strategy: str) -> dict:
        """Runs CV and returns performance metrics for all models."""
        results = {}
        
        # Determine CV strategy
        if validation_strategy == "LOOCV":
            cv = LeaveOneOut()
        elif validation_strategy == "5-Fold CV":
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
        elif "10-Fold CV" in validation_strategy:
            cv = KFold(n_splits=10, shuffle=True, random_state=42)
        else:
            logger.warning(f"Unknown validation strategy: {validation_strategy}. Defaulting to LOOCV.")
            cv = LeaveOneOut()
            
        for target, models in trained_models.items():
            logger.info(f"Evaluating models for target: {target}")
            results[target] = {}
            y_target = y[target].values
            
            for model_name, model in models.items():
                try:
                    # Train predictions (for overfitting detection)
                    y_pred_train = model.predict(X)
                    train_r2 = r2_score(y_target, y_pred_train)
                    
                    # CV predictions
                    y_pred_cv = cross_val_predict(model, X, y_target, cv=cv)
                    
                    cv_r2 = r2_score(y_target, y_pred_cv)
                    cv_rmse = np.sqrt(mean_squared_error(y_target, y_pred_cv))
                    cv_mae = mean_absolute_error(y_target, y_pred_cv)
                    
                    results[target][model_name] = {
                        "Train_R2": train_r2,
                        "CV_R2": cv_r2,
                        "CV_RMSE": cv_rmse,
                        "CV_MAE": cv_mae,
                        "y_pred_cv": y_pred_cv,
                        "y_pred_train": y_pred_train,
                        "model": model # keep reference
                    }
                    logger.info(f"  {model_name}: Train R2={train_r2:.3f}, CV R2={cv_r2:.3f}, RMSE={cv_rmse:.3f}")
                except Exception as e:
                    logger.error(f"  Failed to evaluate {model_name}: {e}")
                    
        return results

    def select_best_models(self, evaluation_results: dict) -> dict:
        """Selects the best model for each target based on CV_R2, avoiding severe overfitting."""
        best_models = {}
        
        for target, models_eval in evaluation_results.items():
            best_model_name = None
            best_score = -float('inf')
            
            for name, metrics in models_eval.items():
                train_r2 = metrics["Train_R2"]
                cv_r2 = metrics["CV_R2"]
                
                # Check for severe overfitting
                is_overfit = (train_r2 - cv_r2) > 0.4 and cv_r2 < 0.5
                
                if not is_overfit and cv_r2 > best_score:
                    best_score = cv_r2
                    best_model_name = name
                    
            if best_model_name is None and models_eval:
                # Fallback to the highest CV_R2 regardless of overfitting if all are overfit
                best_model_name = max(models_eval.items(), key=lambda x: x[1]["CV_R2"])[0]
                
            if best_model_name:
                best_models[target] = models_eval[best_model_name]
                best_models[target]["name"] = best_model_name
                logger.info(f"Selected {best_model_name} as best model for {target} (CV R2: {best_models[target]['CV_R2']:.3f})")
                
        return best_models
