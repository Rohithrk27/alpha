import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import setup_logger

logger = setup_logger(__name__)

class VisualizationEngine:
    """Generates publication-quality visualizations for model performance."""
    
    def __init__(self):
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            pass
            
    def plot_model_comparison(self, evaluation_results: dict, target: str) -> plt.Figure:
        """Plots CV_R2 for all models for a specific target."""
        if target not in evaluation_results:
            return None
            
        models_data = evaluation_results[target]
        if not models_data:
            return None
            
        names = []
        r2_scores = []
        
        for name, metrics in models_data.items():
            names.append(name)
            r2_scores.append(metrics["CV_R2"])
            
        df_plot = pd.DataFrame({'Model': names, 'CV R²': r2_scores})
        df_plot = df_plot.sort_values('CV R²', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color coding: Green > 0.7, Yellow > 0.4, Red < 0.4
        colors = ['#ef4444' if val < 0.4 else '#eab308' if val < 0.7 else '#22c55e' for val in df_plot['CV R²']]
        
        bars = ax.barh(df_plot['Model'], df_plot['CV R²'], color=colors)
        
        ax.set_title(f"Model Comparison - {target} (Cross-Validated R²)", fontsize=14, pad=15)
        ax.set_xlabel("Cross-Validated R² Score", fontsize=12)
        ax.set_xlim(min(-0.2, df_plot['CV R²'].min() - 0.1), 1.0)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.02 if width > 0 else width - 0.05
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                    va='center', fontsize=10)
                    
        fig.tight_layout()
        return fig

    def plot_actual_vs_predicted(self, best_models: dict, y_actual: pd.DataFrame) -> plt.Figure:
        """Plots actual vs predicted values for the best models."""
        n_targets = len(best_models)
        if n_targets == 0:
            return None
            
        fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 5))
        if n_targets == 1:
            axes = [axes]
            
        for ax, (target, model_info) in zip(axes, best_models.items()):
            y_true = y_actual[target].values
            y_pred = model_info["y_pred_train"]
            
            r2 = model_info["Train_R2"]
            rmse = model_info["CV_RMSE"] # Keep CV RMSE for reference if needed
            
            ax.scatter(y_true, y_pred, alpha=0.7, color='#3b82f6', edgecolors='w', s=80)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8)
            
            ax.set_title(f"{target} (Best: {model_info['name']})\nTrain R²={r2:.2f}, CV RMSE={rmse:.2f}")
            ax.set_xlabel("Actual Value")
            ax.set_ylabel("Predicted Value (Training)")
            
        fig.tight_layout()
        return fig

    def plot_chemical_space(self, X_train: pd.DataFrame, X_novel: pd.DataFrame, train_labels: list, novel_labels: list) -> plt.Figure:
        """Plots a 2D PCA representation of the chemical space."""
        from sklearn.decomposition import PCA
        
        if X_train.empty:
            return None
            
        pca = PCA(n_components=2)
        
        try:
            train_pca = pca.fit_transform(X_train)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Plot training data
            ax.scatter(train_pca[:, 0], train_pca[:, 1], c='#3b82f6', label='Training Solvents (Known Space)', alpha=0.6, s=80, edgecolors='w')
            
            # Plot novel data if available
            if not X_novel.empty:
                missing_cols = set(X_train.columns) - set(X_novel.columns)
                for c in missing_cols:
                    X_novel[c] = 0
                X_novel_aligned = X_novel[X_train.columns]
                
                novel_pca = pca.transform(X_novel_aligned)
                ax.scatter(novel_pca[:, 0], novel_pca[:, 1], c='#ef4444', label='Novel Solvents (Predicted)', alpha=0.9, s=200, marker='*', edgecolors='w')
                
                for i, label in enumerate(novel_labels):
                    ax.annotate(label, (novel_pca[i, 0], novel_pca[i, 1]), 
                               xytext=(8, 8), textcoords='offset points', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                               
            ax.set_title("Chemical Space Mapping (PCA of Molecular Descriptors)", fontsize=14, pad=15)
            ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
            ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
            ax.legend()
            
            fig.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Error generating PCA plot: {str(e)}")
            return None

    def plot_prediction_bar_charts(self, results_df: pd.DataFrame) -> plt.Figure:
        """Plots a bar chart with error bars for each target variable."""
        # Find all prediction columns
        target_preds = [c for c in results_df.columns if str(c).endswith("_Prediction")]
        if not target_preds:
            return None
            
        # Only plot the Top 20 solvents to keep the bar chart clean and focused
        plot_df = results_df.head(20).copy()
        
        n_targets = len(target_preds)
        fig, axes = plt.subplots(n_targets, 1, figsize=(14, 7 * n_targets))
        
        # Ensure axes is iterable even if there's only 1 target
        if n_targets == 1:
            axes = [axes]
            
        solvents = plot_df["solvent_name"].tolist()
        x_pos = np.arange(len(solvents))
        
        for ax, pred_col in zip(axes, target_preds):
            target_name = pred_col.replace("_Prediction", "")
            unc_col = f"{target_name}_Uncertainty"
            
            y_vals = plot_df[pred_col].values
            
            if unc_col in plot_df.columns:
                y_err = plot_df[unc_col].values
            else:
                y_err = np.zeros(len(y_vals))
                
            bars = ax.bar(x_pos, y_vals, yerr=y_err, capsize=5, color='#3b82f6', alpha=0.8, edgecolor='black')
            
            # Formatting
            title_text = f"Top {len(solvents)} Solvents: {target_name.replace('_', ' ').title()}"
            ax.set_title(title_text, fontsize=18, pad=20, fontweight='bold', color='#1e3a8a')
            ax.set_ylabel(f"Predicted Value", fontsize=14)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(solvents, rotation=45, ha='right', fontsize=12)
            
            # Add value labels slightly above the error bars for clear readability
            max_y = ax.get_ylim()[1]
            for i, v in enumerate(y_vals):
                label_y = v + y_err[i] + (max_y * 0.02)
                ax.text(i, label_y, f"{v:.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
                
            # Expand the top limit slightly so the text doesn't hit the ceiling
            ax.set_ylim(0, max_y * 1.15)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
        fig.tight_layout()
        return fig

