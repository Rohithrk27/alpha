import pandas as pd
import numpy as np
from src.utils import setup_logger

logger = setup_logger(__name__)

# ── HARDCODED BIOLOGICAL DESCRIPTOR PRIORITY LIST ───────────────────────────
# These are the 4 descriptors explicitly chosen by the researcher based on
# biological importance for predicting whole-cell Pichia kudriavzevii response.
# Ranked: LogP > Dielectric Constant > Water Solubility > Dynamic Viscosity
MANDATED_DESCRIPTORS = [
    ['xlogp', 'logp'],          # Rank 1: Membrane partitioning (lipophilicity)
    ['dielectric_constant'],    # Rank 2: Solvent polarity
    ['water_solubility'],       # Rank 3: Aqueous-phase partitioning
    ['dynamic_viscosity', 'viscosity'],  # Rank 4: Mass transfer / diffusivity
]

class FeatureSelector:
    """Selects the 4 mandated biological descriptors for model training.
    
    For small datasets (<20 rows), the selector is hardcoded to the 4
    researcher-specified descriptors to prevent overfitting and ensure
    biological interpretability. For larger datasets, standard correlation-
    based pruning is applied after the mandated descriptors are secured.
    """
    
    def __init__(self, correlation_threshold: float = 0.9):
        self.correlation_threshold = correlation_threshold
        self.selected_features = []
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature selection...")
        X_new = X.copy()
        
        # Remove constant features first
        constant_cols = [col for col in X_new.columns if X_new[col].nunique() <= 1]
        X_new.drop(columns=constant_cols, inplace=True)
        logger.info(f"Dropped {len(constant_cols)} constant features.")
        
        n_samples = len(X_new)

        # ── MANDATED SELECTION (always applied, regardless of dataset size) ──
        # Walk the priority list in rank order. For each rank group, find the
        # first matching column in X_new (case-insensitive). Collect exactly
        # one representative per rank until all 4 ranks are satisfied.
        cols_to_keep = []
        missing_ranks = []
        for rank_aliases in MANDATED_DESCRIPTORS:
            found = None
            for alias in rank_aliases:
                matches = [c for c in X_new.columns if c.lower() == alias]
                if matches:
                    found = matches[0]
                    break
            if found:
                cols_to_keep.append(found)
            else:
                missing_ranks.append(rank_aliases[0])

        if missing_ranks:
            logger.warning(
                f"Could not find the following mandated descriptors in the data: "
                f"{missing_ranks}. They may not have been fetched from PubChem or "
                f"the offline database. Check that descriptor fetch succeeded."
            )

        if not cols_to_keep:
            # Absolute fallback — should never happen in normal use
            logger.error("Zero mandated descriptors found. Falling back to first 2 numeric columns.")
            cols_to_keep = X_new.select_dtypes(include=['number']).columns.tolist()[:2]

        # Drop everything that is not in the mandated set
        cols_to_drop = [c for c in X_new.columns if c not in cols_to_keep]
        X_new.drop(columns=cols_to_drop, inplace=True)

        found_str = ", ".join(cols_to_keep)
        logger.info(
            f"Feature selection complete. Using {len(cols_to_keep)} mandated "
            f"biological descriptors: [{found_str}]"
        )

        self.selected_features = X_new.columns.tolist()
        return X_new

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features:
            raise ValueError("Selector has not been fitted yet. Call fit_transform first.")
        
        # Only keep selected features that are actually present in X
        cols_to_keep = [col for col in self.selected_features if col in X.columns]
        if not cols_to_keep:
            raise ValueError(
                f"None of the trained features {self.selected_features} are present "
                f"in the prediction data. Ensure descriptor fetch completed successfully."
            )
        return X[cols_to_keep].copy()
