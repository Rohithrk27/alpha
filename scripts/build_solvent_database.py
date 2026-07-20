import os
import sys
import pandas as pd
import time
import logging

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.descriptor_engine import DescriptorEngine

# Configure basic logging for script
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    # A comprehensive curated list of ~150 common organic solvents across various chemical classes
    common_solvents = [
        # Water
        "Water",
        
        # Alcohols
        "Methanol", "Ethanol", "1-Propanol", "2-Propanol", "1-Butanol", "2-Butanol", 
        "Isobutanol", "tert-Butanol", "1-Pentanol", "2-Pentanol", "3-Pentanol", 
        "1-Hexanol", "1-Heptanol", "1-Octanol", "1-Nonanol", "1-Decanol", 
        "Cyclopentanol", "Cyclohexanol", "Benzyl alcohol", "Phenol", 
        "Ethylene glycol", "Propylene glycol", "Glycerol", "Allyl alcohol",
        
        # Ketones
        "Acetone", "Methyl ethyl ketone", "2-Pentanone", "3-Pentanone", 
        "Methyl isobutyl ketone", "Cyclopentanone", "Cyclohexanone", 
        "Acetophenone", "Propiophenone", "2-Hexanone", "Diisobutyl ketone",
        
        # Aldehydes
        "Formaldehyde", "Acetaldehyde", "Propionaldehyde", "Butyraldehyde", "Benzaldehyde",
        
        # Ethers
        "Diethyl ether", "Dipropyl ether", "Diisopropyl ether", "Dibutyl ether", 
        "Methyl tert-butyl ether", "Cyclopentyl methyl ether", "Anisole", 
        "Tetrahydrofuran", "2-Methyltetrahydrofuran", "1,4-Dioxane", 
        "1,2-Dimethoxyethane", "Diethylene glycol dimethyl ether",
        
        # Esters
        "Methyl acetate", "Ethyl acetate", "Propyl acetate", "Isopropyl acetate", 
        "Butyl acetate", "Isobutyl acetate", "tert-Butyl acetate", 
        "Amyl acetate", "Ethyl propionate", "Ethyl butyrate", "Methyl benzoate", 
        "Ethyl benzoate", "Gamma-butyrolactone", "Ethylene carbonate", "Propylene carbonate",
        
        # Alkanes & Cycloalkanes
        "Pentane", "Hexane", "Heptane", "Octane", "Nonane", "Decane", "Dodecane", 
        "Cyclopentane", "Cyclohexane", "Methylcyclohexane", "Isooctane",
        
        # Aromatic Hydrocarbons
        "Benzene", "Toluene", "o-Xylene", "m-Xylene", "p-Xylene", "Ethylbenzene", 
        "Cumene", "Mesitylene", "Styrene", "Chlorobenzene", "o-Dichlorobenzene",
        
        # Halogenated Hydrocarbons
        "Dichloromethane", "Chloroform", "Carbon tetrachloride", "1,2-Dichloroethane", 
        "1,1,1-Trichloroethane", "Trichloroethylene", "Tetrachloroethylene", 
        "1-Chlorobutane", "Fluorobenzene", "Trifluorotoluene",
        
        # Amines & Amides
        "Methylamine", "Ethylamine", "Propylamine", "Butylamine", 
        "Diethylamine", "Triethylamine", "Diisopropylamine", "Aniline", 
        "Pyridine", "Piperidine", "Pyrrolidine", "Morpholine",
        "Formamide", "N,N-Dimethylformamide", "N,N-Dimethylacetamide", 
        "N-Methyl-2-pyrrolidone",
        
        # Nitriles
        "Acetonitrile", "Propionitrile", "Butyronitrile", "Benzonitrile",
        
        # Sulfoxides & Sulfones
        "Dimethyl sulfoxide", "Sulfolane",
        
        # Carboxylic Acids
        "Formic acid", "Acetic acid", "Propionic acid", "Butyric acid", 
        "Valeric acid", "Hexanoic acid", "Trifluoroacetic acid",
        
        # Nitro compounds
        "Nitromethane", "Nitroethane", "Nitrobenzene",
        
        # Miscellaneous
        "Carbon disulfide", "Dimethyl carbonate", "Hexamethylphosphoramide"
    ]
    
    # Remove duplicates and sort
    common_solvents = sorted(list(set(common_solvents)))
    
    logging.info(f"Initialized list of {len(common_solvents)} common solvents.")
    
    # Convert to DataFrame
    df = pd.DataFrame({"solvent_name": common_solvents})
    
    # Instantiate Descriptor Engine
    engine = DescriptorEngine()
    
    logging.info("Starting descriptor fetch (this will take a few minutes to respect API rate limits)...")
    
    # Enrich dataset (DescriptorEngine already handles sleeping/rate-limits internally per batch, 
    # but we are just sending the whole dataframe to it. It uses apply(), which we might want to slow down 
    # if it's large, but 120 items usually pass PubChem limits if we aren't doing it constantly).
    try:
        enriched_df = engine.enrich_dataset(df, "solvent_name")
        
        # Rename solvent_name to match schema
        enriched_df = enriched_df.rename(columns={"solvent_name": "Solvent_Name"})
        
        # Count failed fetches using SMILES before filtering
        failed_count = len(enriched_df[enriched_df['SMILES'] == '']) if 'SMILES' in enriched_df.columns else 0
        
        # Enforce Master Schema Column Order and create placeholders
        master_schema = [
            "Solvent_Name",
            "PubChem_CID",
            "PubChem_CommonName",
            "IUPACName",
            "Molecular_Formula",
            "SMILES",
            "Molecular_Weight",
            "XLogP",
            "Topological_Polar_Surface_Area",
            "Hydrogen_Bond_Donor_Count",
            "Hydrogen_Bond_Acceptor_Count",
            "Rotatable_Bond_Count",
            "Heavy_Atom_Count",
            "Aromatic_Ring_Count",
            "Fraction_CSP3",
            "Density",
            "Dynamic_Viscosity",
            "Dielectric_Constant",
            "Surface_Tension",
            "Vapor_Pressure",
            "Boiling_Point",
            "Water_Solubility",
            "Hansen_Dispersion",
            "Hansen_Polar",
            "Hansen_Hydrogen",
            "Solvent_Concentration"
        ]
        
        # Add any missing placeholder columns
        for col in master_schema:
            if col not in enriched_df.columns:
                enriched_df[col] = pd.NA
                
        # Reorder and filter columns precisely to master schema
        enriched_df = enriched_df[master_schema]
        
        # Ensure data directory exists
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save to CSV
        output_path = os.path.join(data_dir, 'universal_solvent_database.csv')
        enriched_df.to_csv(output_path, index=False)
        
        logging.info(f"Successfully generated database at: {output_path}")
        logging.info(f"Total solvents processed: {len(enriched_df)}")
        logging.info(f"Failed fetches (if any): {failed_count}")
        
    except Exception as e:
        logging.error(f"Failed to build database: {e}")

if __name__ == "__main__":
    main()
