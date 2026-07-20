import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import setup_logger

logger = setup_logger(__name__)

class ExploratoryAnalysis:
    """Generates visualizations and correlation matrices for EDA."""
    
    def __init__(self):
        # Apply standard style without using context managers
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            pass # Fallback to default if style not found
        
    def generate_correlation_matrix(self, df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
        """Returns the correlation of numerical features with the targets."""
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) == 0:
            return pd.DataFrame()
            
        corr = df[num_cols].corr()
        
        # Extract correlations with targets
        target_corr = corr.loc[[c for c in target_cols if c in corr.index], 
                                [c for c in corr.columns if c not in target_cols]]
        
        return target_corr.T.sort_values(by=target_cols[0], ascending=False, key=abs) if not target_corr.empty else pd.DataFrame()

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> plt.Figure:
        """Plots a correlation heatmap for numerical features."""
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough numerical features", ha='center')
            return fig
            
        corr = df[num_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        fig.tight_layout()
        return fig
        
    def plot_target_distributions(self, df: pd.DataFrame, target_cols: list[str]) -> plt.Figure:
        """Plots the distribution of the target variables."""
        n_targets = len(target_cols)
        if n_targets == 0:
            fig, ax = plt.subplots()
            return fig
            
        fig, axes = plt.subplots(1, n_targets, figsize=(5*n_targets, 4))
        if n_targets == 1:
            axes = [axes]
            
        for ax, col in zip(axes, target_cols):
            if col in df.columns:
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                
        fig.tight_layout()
        return fig
