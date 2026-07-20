import time
import pandas as pd
import requests
from src.utils import setup_logger

logger = setup_logger(__name__)

class DescriptorEngine:
    """Fetches molecular descriptors from PubChem and computes them via RDKit."""
    
    PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    PUBCHEM_PROPERTIES = [
        "MolecularWeight", "XLogP", "TPSA", "HBondDonorCount", 
        "HBondAcceptorCount", "RotatableBondCount", "Complexity", 
        "ExactMass", "MolecularFormula", "IsomericSMILES", "IUPACName", "Title"
    ]
    
    def fetch_pubchem_properties(self, solvent_name: str, max_retries: int = 3) -> dict | None:
        """Fetches basic descriptors and SMILES from PubChem."""
        if solvent_name.lower().startswith("control"):
            return None

        props_str = ",".join(self.PUBCHEM_PROPERTIES)
        url = f"{self.PUBCHEM_BASE}/compound/name/{solvent_name}/property/{props_str}/JSON"

        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    props = data["PropertyTable"]["Properties"][0]
                    return {
                        "PubChem_CID": props.get("CID"),
                        "PubChem_CommonName": props.get("Title", ""),
                        "IUPACName": props.get("IUPACName", ""),
                        "SMILES": props.get("IsomericSMILES", props.get("SMILES", "")),
                        "Molecular_Formula": props.get("MolecularFormula", ""),
                        "Molecular_Weight": float(props.get("MolecularWeight", 0)),
                        "XLogP": props.get("XLogP"),
                        "Topological_Polar_Surface_Area": float(props.get("TPSA", 0)),
                        "Hydrogen_Bond_Donor_Count": int(props.get("HBondDonorCount", 0)),
                        "Hydrogen_Bond_Acceptor_Count": int(props.get("HBondAcceptorCount", 0)),
                        "Rotatable_Bond_Count": int(props.get("RotatableBondCount", 0)),
                    }
                elif resp.status_code == 404:
                    logger.warning(f"'{solvent_name}' not found in PubChem.")
                    return None
                else:
                    logger.warning(f"PubChem returned {resp.status_code} for '{solvent_name}', retrying...")
                    time.sleep(1)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error for '{solvent_name}': {e}, retrying...")
                time.sleep(2)

        logger.error(f"Failed to fetch '{solvent_name}' after {max_retries} attempts.")
        return None

    def compute_rdkit_descriptors(self, smiles: str) -> dict:
        """Computes additional descriptors using RDKit if available."""
        if not smiles or pd.isna(smiles):
            return {}
            
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
                
            return {
                "Heavy_Atom_Count": Descriptors.HeavyAtomCount(mol),
                "Aromatic_Ring_Count": Descriptors.NumAromaticRings(mol),
                "Fraction_CSP3": Descriptors.FractionCSP3(mol),
            }
        except ImportError:
            logger.info("RDKit not installed. Skipping RDKit descriptors.")
            return {}

    def enrich_dataset(self, df: pd.DataFrame, solvent_col: str) -> pd.DataFrame:
        """Enriches the dataframe with all available descriptors."""
        logger.info("Starting descriptor collection...")
        all_descriptors = []
        
        for idx, row in df.iterrows():
            solvent = row[solvent_col]
            desc = {solvent_col: solvent}
            
            pubchem_data = self.fetch_pubchem_properties(solvent)
            if pubchem_data:
                smiles = pubchem_data.pop("SMILES", "")
                desc.update(pubchem_data)
                desc["SMILES"] = smiles
                
                mol_desc = self.compute_rdkit_descriptors(smiles)
                desc.update(mol_desc)
                logger.info(f"Processed: {solvent}")
            else:
                logger.info(f"Skipped: {solvent}")
                
            all_descriptors.append(desc)
            time.sleep(0.3)  # Rate limiting
            
        desc_df = pd.DataFrame(all_descriptors)
        
        if not desc_df.empty:
            # Avoid _x and _y column duplication by combining data smartly
            df_indexed = df.set_index(solvent_col)
            desc_indexed = desc_df.set_index(solvent_col)
            
            # combine_first prioritizes non-null values from existing df, filling gaps with newly fetched data
            merged_df = df_indexed.combine_first(desc_indexed).reset_index()
        else:
            merged_df = df
            
        logger.info(f"Completed descriptor collection.")
        return merged_df
