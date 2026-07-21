# Solvent Toxicity Prediction Framework for Whole-Cell *Pichia kudriavzevii*

## Overview
A hybrid experimental–machine learning framework capable of predicting the toxicity of organic solvents toward whole-cell *Pichia kudriavzevii* based on experimentally measured responses (e.g., Glucose Uptake, Substrate Conversion) and solvent physicochemical descriptors. 

The software is intentionally designed to scale from small proof-of-concept datasets to large experimental and literature-derived datasets without requiring structural modifications, using a dynamic **Model Recommendation Engine**.

For a conceptual, non-technical breakdown of how this AI framework works and why it is highly relevant to modern biotechnology and wet-lab research, please read the [Biotech Explanation Document](explanation.md).

## Architecture (Version 3.0)
The system operates on a highly modular architecture located in `src/`:
- **`data_manager.py`**: Experimental dataset loading, handling dynamically scaling numbers of biological targets.
- **`descriptor_engine.py`**: Automated molecular property fetching via PubChem and RDKit.
- **`recommendation_engine.py`**: Dynamically recommends ML models (e.g., heavily prioritizing Gaussian Process regressors for tiny biological lab datasets).
- **`model_training.py` & `model_validation.py`**: Concurrent multi-target regression scaling to an unlimited amount of input columns.
- **`prediction_engine.py`**: Multi-target prediction inference for 150+ novel solvents with calculated 95% Confidence Intervals.
- **`visualization.py`**: Generates publication-ready chemical space PCA plots, Min-Max normalized Overall Compatibility scores, dynamic Bar Charts with error bars, and 2D RDKit structural renderings.
- **`model_interpretation.py`**: Explainable AI via SHAP summaries.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

*(Note: Install `xgboost`, `catboost`, `lightgbm`, and `shap` if you wish to utilize the advanced models and Explainable AI features recommended for larger datasets. Ensure `rdkit` is installed to view 2D molecular structure drawings.)*

## Universal Solvent Database
The framework includes a script to build a comprehensive offline descriptor database for ~150 common industrial and laboratory organic solvents. This helps avoid hitting API rate limits during massive continuous predictions.
To generate this database, run:
```bash
python scripts/build_solvent_database.py
```
This will output `data/universal_solvent_database.csv`, containing exactly 26 columns (6 identification/structure columns and 20 descriptor/property factors):
- **Identification & Structure**: `Solvent_Name`, `PubChem_CID`, `PubChem_CommonName`, `IUPACName`, `Molecular_Formula`, `SMILES`
- **Computed Descriptors (PubChem/RDKit)**: `Molecular_Weight`, `XLogP`, `Topological_Polar_Surface_Area`, `Hydrogen_Bond_Donor_Count`, `Hydrogen_Bond_Acceptor_Count`, `Rotatable_Bond_Count`, `Heavy_Atom_Count`, `Aromatic_Ring_Count`, `Fraction_CSP3`
- **Physicochemical Properties**: `Density`, `Dynamic_Viscosity`, `Dielectric_Constant`, `Surface_Tension`, `Vapor_Pressure`, `Boiling_Point`, `Water_Solubility`
- **Hansen Parameters**: `Hansen_Dispersion`, `Hansen_Polar`, `Hansen_Hydrogen`
- **Experimental Factor**: `Solvent_Concentration`

## Usage

The entire framework is accessible via an interactive Streamlit application.

```bash
streamlit run app.py
```

### Application Workflow:
1. **Phase 1: Data Initialization & Validation**: Upload your experimental CSV, attach metadata (Media, Temp, etc.), and use the interactive Data Editor to manually inject missing descriptors or fetch missing structural descriptors via PubChem.
2. **Phase 2: Model Training & Evaluation**: The engine analyzes dataset shape, recommends models (defaulting to Gaussian Process for small datasets), and trains them across all targets concurrently.
3. **Phase 3: High-Throughput Screening**: Evaluates a library of 150+ untested chemicals. Automatically generates 2D structural graphs, PCA visualizations, and mathematically scaled Bar Charts with 95% Confidence Intervals pointing you toward optimal biocompatible chemicals.

## Data Structure
Your experimental data can be provided via a CSV file, or typed and edited manually row-by-row within the application interface. The software dynamically handles any number of target columns you provide.
Example:
| solvent_name | glucose_uptake | substrate_conversion | cfu/ml | any_other_target |
|--------------|----------------|----------------------|--------|------------------|
| Ethanol      | 78             | 71                   | 500000 | ...              |
| Acetone      | 60             | 55                   | 10000  | ...              |
| Toluene      | 18             | 12                   | 0      | ...              |
