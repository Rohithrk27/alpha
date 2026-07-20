"""
fill_database.py - Comprehensive database enrichment script.

Fills empty columns in universal_solvent_database.csv using:
1. PubChem PUG REST API (HeavyAtomCount + experimental properties)
2. Curated literature data (Hansen parameters, density, boiling point,
   viscosity, dielectric constant, surface tension, vapor pressure,
   water solubility, aromatic ring count, fraction CSP3)

Sources: CRC Handbook of Chemistry and Physics, Hansen Solubility Parameters
(2nd Ed.), Perry's Chemical Engineers' Handbook, PubChem.
"""

import os
import sys
import time
import json
import logging
import requests
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated literature data for common solvents
# Keys are LOWER-CASED solvent names for matching.
# Values: dict of {column_name: value}
# ---------------------------------------------------------------------------

# fmt: off
LITERATURE = {
    # ── Water ──
    "water": {"Boiling_Point": 100.0, "Density": 0.998, "Dynamic_Viscosity": 1.002, "Dielectric_Constant": 80.1, "Surface_Tension": 72.8, "Vapor_Pressure": 2.34, "Water_Solubility": 1e6, "Hansen_Dispersion": 15.5, "Hansen_Polar": 16.0, "Hansen_Hydrogen": 42.3, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.0},

    # ── Alcohols ──
    "methanol": {"Boiling_Point": 64.7, "Density": 0.791, "Dynamic_Viscosity": 0.544, "Dielectric_Constant": 32.7, "Surface_Tension": 22.1, "Vapor_Pressure": 13.02, "Water_Solubility": 1e6, "Hansen_Dispersion": 15.1, "Hansen_Polar": 12.3, "Hansen_Hydrogen": 22.3, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "ethanol": {"Boiling_Point": 78.4, "Density": 0.789, "Dynamic_Viscosity": 1.074, "Dielectric_Constant": 24.6, "Surface_Tension": 22.1, "Vapor_Pressure": 5.95, "Water_Solubility": 1e6, "Hansen_Dispersion": 15.8, "Hansen_Polar": 8.8, "Hansen_Hydrogen": 19.4, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "1-propanol": {"Boiling_Point": 97.2, "Density": 0.804, "Dynamic_Viscosity": 1.945, "Dielectric_Constant": 20.3, "Surface_Tension": 23.3, "Vapor_Pressure": 1.99, "Water_Solubility": 1e6, "Hansen_Dispersion": 16.0, "Hansen_Polar": 6.8, "Hansen_Hydrogen": 17.4, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "2-propanol": {"Boiling_Point": 82.6, "Density": 0.786, "Dynamic_Viscosity": 2.038, "Dielectric_Constant": 19.9, "Surface_Tension": 21.7, "Vapor_Pressure": 4.4, "Water_Solubility": 1e6, "Hansen_Dispersion": 15.8, "Hansen_Polar": 6.1, "Hansen_Hydrogen": 16.4, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "1-butanol": {"Boiling_Point": 117.7, "Density": 0.810, "Dynamic_Viscosity": 2.544, "Dielectric_Constant": 17.8, "Surface_Tension": 24.6, "Vapor_Pressure": 0.58, "Water_Solubility": 73000, "Hansen_Dispersion": 16.0, "Hansen_Polar": 5.7, "Hansen_Hydrogen": 15.8, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "2-butanol": {"Boiling_Point": 99.5, "Density": 0.808, "Dynamic_Viscosity": 3.096, "Dielectric_Constant": 16.6, "Surface_Tension": 23.5, "Vapor_Pressure": 1.67, "Water_Solubility": 181000, "Hansen_Dispersion": 15.8, "Hansen_Polar": 5.7, "Hansen_Hydrogen": 14.5, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "isobutanol": {"Boiling_Point": 108.0, "Density": 0.802, "Dynamic_Viscosity": 3.33, "Dielectric_Constant": 17.9, "Surface_Tension": 22.9, "Vapor_Pressure": 1.21, "Water_Solubility": 85000, "Hansen_Dispersion": 15.1, "Hansen_Polar": 5.7, "Hansen_Hydrogen": 16.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "tert-butanol": {"Boiling_Point": 82.4, "Density": 0.789, "Dynamic_Viscosity": 3.35, "Dielectric_Constant": 12.5, "Surface_Tension": 20.4, "Vapor_Pressure": 4.1, "Water_Solubility": 1e6, "Hansen_Dispersion": 15.2, "Hansen_Polar": 5.1, "Hansen_Hydrogen": 14.7, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "1-pentanol": {"Boiling_Point": 138.0, "Density": 0.814, "Dynamic_Viscosity": 3.619, "Dielectric_Constant": 13.9, "Surface_Tension": 25.4, "Vapor_Pressure": 0.259, "Water_Solubility": 22000, "Hansen_Dispersion": 15.9, "Hansen_Polar": 4.5, "Hansen_Hydrogen": 13.9, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "2-pentanol": {"Boiling_Point": 119.0, "Density": 0.809, "Dynamic_Viscosity": 3.47, "Dielectric_Constant": 13.1, "Surface_Tension": 23.5, "Vapor_Pressure": 0.76, "Water_Solubility": 45000, "Hansen_Dispersion": 15.6, "Hansen_Polar": 5.0, "Hansen_Hydrogen": 13.3, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "3-pentanol": {"Boiling_Point": 115.5, "Density": 0.821, "Dynamic_Viscosity": 3.35, "Dielectric_Constant": 13.4, "Surface_Tension": 24.0, "Vapor_Pressure": 0.93, "Water_Solubility": 55000, "Hansen_Dispersion": 15.8, "Hansen_Polar": 5.2, "Hansen_Hydrogen": 13.1, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "1-hexanol": {"Boiling_Point": 157.6, "Density": 0.814, "Dynamic_Viscosity": 4.578, "Dielectric_Constant": 13.0, "Surface_Tension": 25.8, "Vapor_Pressure": 0.093, "Water_Solubility": 5900, "Hansen_Dispersion": 15.9, "Hansen_Polar": 5.8, "Hansen_Hydrogen": 12.5, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "1-heptanol": {"Boiling_Point": 176.0, "Density": 0.822, "Dynamic_Viscosity": 5.81, "Dielectric_Constant": 11.8, "Surface_Tension": 26.7, "Vapor_Pressure": 0.036, "Water_Solubility": 1670, "Hansen_Dispersion": 16.0, "Hansen_Polar": 5.3, "Hansen_Hydrogen": 11.7, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "1-octanol": {"Boiling_Point": 195.0, "Density": 0.824, "Dynamic_Viscosity": 7.288, "Dielectric_Constant": 10.3, "Surface_Tension": 27.5, "Vapor_Pressure": 0.011, "Water_Solubility": 540, "Hansen_Dispersion": 16.0, "Hansen_Polar": 5.0, "Hansen_Hydrogen": 11.2, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "1-nonanol": {"Boiling_Point": 213.0, "Density": 0.827, "Dynamic_Viscosity": 9.12, "Dielectric_Constant": 8.8, "Surface_Tension": 28.3, "Vapor_Pressure": 0.004, "Water_Solubility": 140, "Hansen_Dispersion": 16.0, "Hansen_Polar": 4.6, "Hansen_Hydrogen": 10.7, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "1-decanol": {"Boiling_Point": 233.0, "Density": 0.829, "Dynamic_Viscosity": 11.4, "Dielectric_Constant": 8.1, "Surface_Tension": 28.5, "Vapor_Pressure": 0.001, "Water_Solubility": 37, "Hansen_Dispersion": 16.0, "Hansen_Polar": 4.3, "Hansen_Hydrogen": 10.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "cyclopentanol": {"Boiling_Point": 140.9, "Density": 0.949, "Dynamic_Viscosity": 5.16, "Dielectric_Constant": 18.5, "Surface_Tension": 32.7, "Vapor_Pressure": 0.35, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "cyclohexanol": {"Boiling_Point": 161.1, "Density": 0.962, "Dynamic_Viscosity": 41.07, "Dielectric_Constant": 15.0, "Surface_Tension": 33.4, "Vapor_Pressure": 0.065, "Water_Solubility": 36000, "Hansen_Dispersion": 17.4, "Hansen_Polar": 4.1, "Hansen_Hydrogen": 13.5, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "benzyl alcohol": {"Boiling_Point": 205.3, "Density": 1.044, "Dynamic_Viscosity": 5.47, "Dielectric_Constant": 13.0, "Surface_Tension": 39.0, "Vapor_Pressure": 0.013, "Water_Solubility": 40000, "Hansen_Dispersion": 18.4, "Hansen_Polar": 6.3, "Hansen_Hydrogen": 13.7, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.143},
    "phenol": {"Boiling_Point": 181.7, "Density": 1.071, "Dynamic_Viscosity": 3.437, "Dielectric_Constant": 9.8, "Surface_Tension": 40.9, "Vapor_Pressure": 0.036, "Water_Solubility": 82800, "Hansen_Dispersion": 18.0, "Hansen_Polar": 5.9, "Hansen_Hydrogen": 14.9, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.0},
    "ethylene glycol": {"Boiling_Point": 197.3, "Density": 1.113, "Dynamic_Viscosity": 16.1, "Dielectric_Constant": 37.7, "Surface_Tension": 47.7, "Vapor_Pressure": 0.008, "Water_Solubility": 1e6, "Hansen_Dispersion": 17.0, "Hansen_Polar": 11.0, "Hansen_Hydrogen": 26.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "propylene glycol": {"Boiling_Point": 188.2, "Density": 1.036, "Dynamic_Viscosity": 40.4, "Dielectric_Constant": 32.0, "Surface_Tension": 36.5, "Vapor_Pressure": 0.013, "Water_Solubility": 1e6, "Hansen_Dispersion": 16.8, "Hansen_Polar": 9.4, "Hansen_Hydrogen": 23.3, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "glycerol": {"Boiling_Point": 290.0, "Density": 1.261, "Dynamic_Viscosity": 1412.0, "Dielectric_Constant": 42.5, "Surface_Tension": 63.4, "Vapor_Pressure": 0.00003, "Water_Solubility": 1e6, "Hansen_Dispersion": 17.4, "Hansen_Polar": 12.1, "Hansen_Hydrogen": 29.3, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "allyl alcohol": {"Boiling_Point": 97.1, "Density": 0.854, "Dynamic_Viscosity": 1.218, "Dielectric_Constant": 19.7, "Surface_Tension": 25.3, "Vapor_Pressure": 2.63, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.333},

    # ── Ketones ──
    "acetone": {"Boiling_Point": 56.1, "Density": 0.791, "Dynamic_Viscosity": 0.306, "Dielectric_Constant": 20.7, "Surface_Tension": 25.2, "Vapor_Pressure": 24.6, "Water_Solubility": 1e6, "Hansen_Dispersion": 15.5, "Hansen_Polar": 10.4, "Hansen_Hydrogen": 7.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.667},
    "methyl ethyl ketone": {"Boiling_Point": 79.6, "Density": 0.805, "Dynamic_Viscosity": 0.405, "Dielectric_Constant": 18.5, "Surface_Tension": 24.6, "Vapor_Pressure": 10.5, "Water_Solubility": 223000, "Hansen_Dispersion": 16.0, "Hansen_Polar": 9.0, "Hansen_Hydrogen": 5.1, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.75},
    "2-pentanone": {"Boiling_Point": 102.0, "Density": 0.809, "Dynamic_Viscosity": 0.473, "Dielectric_Constant": 15.4, "Surface_Tension": 25.0, "Vapor_Pressure": 3.37, "Water_Solubility": 43000, "Hansen_Dispersion": 15.8, "Hansen_Polar": 7.6, "Hansen_Hydrogen": 4.7, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.8},
    "3-pentanone": {"Boiling_Point": 102.0, "Density": 0.814, "Dynamic_Viscosity": 0.444, "Dielectric_Constant": 17.0, "Surface_Tension": 24.7, "Vapor_Pressure": 3.53, "Water_Solubility": 34000, "Hansen_Dispersion": 15.8, "Hansen_Polar": 7.6, "Hansen_Hydrogen": 4.7, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.8},
    "methyl isobutyl ketone": {"Boiling_Point": 117.0, "Density": 0.801, "Dynamic_Viscosity": 0.585, "Dielectric_Constant": 13.1, "Surface_Tension": 23.6, "Vapor_Pressure": 2.09, "Water_Solubility": 19000, "Hansen_Dispersion": 15.3, "Hansen_Polar": 6.1, "Hansen_Hydrogen": 4.1, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.833},
    "cyclopentanone": {"Boiling_Point": 130.6, "Density": 0.949, "Dynamic_Viscosity": 1.07, "Dielectric_Constant": 13.6, "Surface_Tension": 33.3, "Vapor_Pressure": 1.2, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.8},
    "cyclohexanone": {"Boiling_Point": 155.6, "Density": 0.947, "Dynamic_Viscosity": 2.02, "Dielectric_Constant": 18.3, "Surface_Tension": 34.4, "Vapor_Pressure": 0.53, "Water_Solubility": 23000, "Hansen_Dispersion": 17.8, "Hansen_Polar": 6.3, "Hansen_Hydrogen": 5.1, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.833},
    "acetophenone": {"Boiling_Point": 202.0, "Density": 1.028, "Dynamic_Viscosity": 1.681, "Dielectric_Constant": 17.4, "Surface_Tension": 39.8, "Vapor_Pressure": 0.045, "Water_Solubility": 5500, "Hansen_Dispersion": 19.6, "Hansen_Polar": 8.6, "Hansen_Hydrogen": 3.7, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.125},
    "propiophenone": {"Boiling_Point": 218.0, "Density": 1.009, "Dynamic_Viscosity": 1.86, "Dielectric_Constant": 16.0, "Surface_Tension": 36.6, "Vapor_Pressure": 0.013, "Water_Solubility": 1700, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.222},
    "2-hexanone": {"Boiling_Point": 127.0, "Density": 0.811, "Dynamic_Viscosity": 0.583, "Dielectric_Constant": 14.1, "Surface_Tension": 25.2, "Vapor_Pressure": 1.13, "Water_Solubility": 14000, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.833},
    "diisobutyl ketone": {"Boiling_Point": 169.4, "Density": 0.806, "Dynamic_Viscosity": 1.04, "Dielectric_Constant": 9.9, "Surface_Tension": 24.1, "Vapor_Pressure": 0.23, "Water_Solubility": 430, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.889},

    # ── Aldehydes ──
    "formaldehyde": {"Boiling_Point": -19.0, "Density": 0.815, "Dynamic_Viscosity": 0.23, "Dielectric_Constant": 2.8, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.0},
    "acetaldehyde": {"Boiling_Point": 20.2, "Density": 0.785, "Dynamic_Viscosity": 0.215, "Dielectric_Constant": 21.1, "Surface_Tension": 21.2, "Vapor_Pressure": 100.7, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.5},
    "propionaldehyde": {"Boiling_Point": 49.0, "Density": 0.805, "Dynamic_Viscosity": 0.325, "Dielectric_Constant": 18.5, "Vapor_Pressure": 30.7, "Water_Solubility": 306000, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.667},
    "butyraldehyde": {"Boiling_Point": 74.8, "Density": 0.803, "Dynamic_Viscosity": 0.38, "Dielectric_Constant": 13.4, "Vapor_Pressure": 11.6, "Water_Solubility": 71000, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.75},
    "benzaldehyde": {"Boiling_Point": 178.1, "Density": 1.044, "Dynamic_Viscosity": 1.39, "Dielectric_Constant": 17.8, "Surface_Tension": 38.9, "Vapor_Pressure": 0.17, "Water_Solubility": 6950, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.0},

    # ── Ethers ──
    "diethyl ether": {"Boiling_Point": 34.6, "Density": 0.713, "Dynamic_Viscosity": 0.224, "Dielectric_Constant": 4.3, "Surface_Tension": 17.1, "Vapor_Pressure": 58.6, "Water_Solubility": 69000, "Hansen_Dispersion": 14.5, "Hansen_Polar": 2.9, "Hansen_Hydrogen": 5.1, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "dipropyl ether": {"Boiling_Point": 90.0, "Density": 0.736, "Dynamic_Viscosity": 0.396, "Dielectric_Constant": 3.4, "Surface_Tension": 20.5, "Vapor_Pressure": 7.93, "Water_Solubility": 4900, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "diisopropyl ether": {"Boiling_Point": 68.5, "Density": 0.724, "Dynamic_Viscosity": 0.379, "Dielectric_Constant": 3.9, "Surface_Tension": 18.0, "Vapor_Pressure": 15.9, "Water_Solubility": 12000, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "dibutyl ether": {"Boiling_Point": 141.0, "Density": 0.769, "Dynamic_Viscosity": 0.637, "Dielectric_Constant": 3.1, "Surface_Tension": 23.0, "Vapor_Pressure": 0.67, "Water_Solubility": 300, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "methyl tert-butyl ether": {"Boiling_Point": 55.2, "Density": 0.740, "Dynamic_Viscosity": 0.36, "Dielectric_Constant": 2.6, "Surface_Tension": 19.4, "Vapor_Pressure": 27.0, "Water_Solubility": 42000, "Hansen_Dispersion": 14.8, "Hansen_Polar": 4.3, "Hansen_Hydrogen": 5.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "cyclopentyl methyl ether": {"Boiling_Point": 106.0, "Density": 0.860, "Dynamic_Viscosity": 0.55, "Dielectric_Constant": 4.8, "Surface_Tension": 24.6, "Vapor_Pressure": 5.56, "Water_Solubility": 15000, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "anisole": {"Boiling_Point": 154.0, "Density": 0.995, "Dynamic_Viscosity": 1.056, "Dielectric_Constant": 4.3, "Surface_Tension": 35.1, "Vapor_Pressure": 0.47, "Water_Solubility": 1540, "Hansen_Dispersion": 17.8, "Hansen_Polar": 4.1, "Hansen_Hydrogen": 6.7, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.143},
    "tetrahydrofuran": {"Boiling_Point": 66.0, "Density": 0.889, "Dynamic_Viscosity": 0.456, "Dielectric_Constant": 7.6, "Surface_Tension": 26.4, "Vapor_Pressure": 21.6, "Water_Solubility": 1e6, "Hansen_Dispersion": 16.8, "Hansen_Polar": 5.7, "Hansen_Hydrogen": 8.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "2-methyltetrahydrofuran": {"Boiling_Point": 80.3, "Density": 0.855, "Dynamic_Viscosity": 0.46, "Dielectric_Constant": 6.2, "Surface_Tension": 24.8, "Vapor_Pressure": 10.7, "Water_Solubility": 140000, "Hansen_Dispersion": 16.9, "Hansen_Polar": 5.0, "Hansen_Hydrogen": 4.3, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "1,4-dioxane": {"Boiling_Point": 101.1, "Density": 1.033, "Dynamic_Viscosity": 1.177, "Dielectric_Constant": 2.3, "Surface_Tension": 33.0, "Vapor_Pressure": 4.95, "Water_Solubility": 1e6, "Hansen_Dispersion": 19.0, "Hansen_Polar": 1.8, "Hansen_Hydrogen": 7.4, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "1,2-dimethoxyethane": {"Boiling_Point": 85.0, "Density": 0.868, "Dynamic_Viscosity": 0.455, "Dielectric_Constant": 7.2, "Surface_Tension": 24.0, "Vapor_Pressure": 6.72, "Water_Solubility": 1e6, "Hansen_Dispersion": 15.5, "Hansen_Polar": 5.8, "Hansen_Hydrogen": 8.4, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "diethylene glycol dimethyl ether": {"Boiling_Point": 162.0, "Density": 0.943, "Dynamic_Viscosity": 1.089, "Dielectric_Constant": 7.2, "Surface_Tension": 28.8, "Vapor_Pressure": 0.39, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},

    # ── Esters ──
    "methyl acetate": {"Boiling_Point": 56.9, "Density": 0.934, "Dynamic_Viscosity": 0.364, "Dielectric_Constant": 6.7, "Surface_Tension": 24.8, "Vapor_Pressure": 21.6, "Water_Solubility": 319000, "Hansen_Dispersion": 15.5, "Hansen_Polar": 7.2, "Hansen_Hydrogen": 7.6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.5},
    "ethyl acetate": {"Boiling_Point": 77.1, "Density": 0.902, "Dynamic_Viscosity": 0.426, "Dielectric_Constant": 6.0, "Surface_Tension": 23.9, "Vapor_Pressure": 10.0, "Water_Solubility": 83000, "Hansen_Dispersion": 15.8, "Hansen_Polar": 5.3, "Hansen_Hydrogen": 7.2, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.667},
    "propyl acetate": {"Boiling_Point": 101.5, "Density": 0.888, "Dynamic_Viscosity": 0.544, "Dielectric_Constant": 5.5, "Surface_Tension": 24.3, "Vapor_Pressure": 3.33, "Water_Solubility": 18900, "Hansen_Dispersion": 15.3, "Hansen_Polar": 4.3, "Hansen_Hydrogen": 7.6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.75},
    "isopropyl acetate": {"Boiling_Point": 88.6, "Density": 0.874, "Dynamic_Viscosity": 0.476, "Dielectric_Constant": 5.6, "Surface_Tension": 22.3, "Vapor_Pressure": 6.0, "Water_Solubility": 31000, "Hansen_Dispersion": 14.9, "Hansen_Polar": 4.5, "Hansen_Hydrogen": 7.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.75},
    "butyl acetate": {"Boiling_Point": 126.1, "Density": 0.882, "Dynamic_Viscosity": 0.685, "Dielectric_Constant": 5.1, "Surface_Tension": 25.1, "Vapor_Pressure": 1.17, "Water_Solubility": 6800, "Hansen_Dispersion": 15.8, "Hansen_Polar": 3.7, "Hansen_Hydrogen": 6.3, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.833},
    "isobutyl acetate": {"Boiling_Point": 117.0, "Density": 0.871, "Dynamic_Viscosity": 0.67, "Dielectric_Constant": 5.3, "Surface_Tension": 23.7, "Vapor_Pressure": 1.87, "Water_Solubility": 6300, "Hansen_Dispersion": 15.1, "Hansen_Polar": 3.7, "Hansen_Hydrogen": 6.3, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.833},
    "tert-butyl acetate": {"Boiling_Point": 98.0, "Density": 0.866, "Dynamic_Viscosity": 0.62, "Dielectric_Constant": 5.0, "Surface_Tension": 22.4, "Vapor_Pressure": 4.13, "Water_Solubility": 7000, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.833},
    "amyl acetate": {"Boiling_Point": 149.0, "Density": 0.876, "Dynamic_Viscosity": 0.862, "Dielectric_Constant": 4.8, "Surface_Tension": 25.7, "Vapor_Pressure": 0.45, "Water_Solubility": 1730, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.857},
    "ethyl propionate": {"Boiling_Point": 99.0, "Density": 0.891, "Dynamic_Viscosity": 0.501, "Dielectric_Constant": 5.6, "Surface_Tension": 23.8, "Vapor_Pressure": 3.53, "Water_Solubility": 19000, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.75},
    "ethyl butyrate": {"Boiling_Point": 121.0, "Density": 0.879, "Dynamic_Viscosity": 0.639, "Dielectric_Constant": 5.1, "Surface_Tension": 24.5, "Vapor_Pressure": 1.3, "Water_Solubility": 5100, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.833},
    "methyl benzoate": {"Boiling_Point": 199.0, "Density": 1.094, "Dynamic_Viscosity": 1.857, "Dielectric_Constant": 6.6, "Surface_Tension": 37.0, "Vapor_Pressure": 0.052, "Water_Solubility": 2100, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.125},
    "ethyl benzoate": {"Boiling_Point": 213.0, "Density": 1.047, "Dynamic_Viscosity": 2.01, "Dielectric_Constant": 6.0, "Surface_Tension": 35.0, "Vapor_Pressure": 0.023, "Water_Solubility": 710, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.222},
    "gamma-butyrolactone": {"Boiling_Point": 204.0, "Density": 1.129, "Dynamic_Viscosity": 1.73, "Dielectric_Constant": 39.0, "Surface_Tension": 44.6, "Vapor_Pressure": 0.043, "Water_Solubility": 1e6, "Hansen_Dispersion": 19.0, "Hansen_Polar": 16.6, "Hansen_Hydrogen": 7.4, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.75},
    "ethylene carbonate": {"Boiling_Point": 248.0, "Density": 1.321, "Dynamic_Viscosity": 1.93, "Dielectric_Constant": 89.8, "Surface_Tension": 48.0, "Water_Solubility": 1e6, "Hansen_Dispersion": 19.4, "Hansen_Polar": 21.7, "Hansen_Hydrogen": 5.1, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.667},
    "propylene carbonate": {"Boiling_Point": 242.0, "Density": 1.205, "Dynamic_Viscosity": 2.53, "Dielectric_Constant": 64.0, "Surface_Tension": 41.4, "Vapor_Pressure": 0.004, "Water_Solubility": 175000, "Hansen_Dispersion": 20.0, "Hansen_Polar": 18.0, "Hansen_Hydrogen": 4.1, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.75},
    "dimethyl carbonate": {"Boiling_Point": 90.1, "Density": 1.073, "Dynamic_Viscosity": 0.585, "Dielectric_Constant": 3.1, "Surface_Tension": 28.5, "Vapor_Pressure": 5.7, "Water_Solubility": 139000, "Hansen_Dispersion": 15.5, "Hansen_Polar": 8.6, "Hansen_Hydrogen": 9.7, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.333},

    # ── Alkanes & Cycloalkanes ──
    "pentane": {"Boiling_Point": 36.1, "Density": 0.626, "Dynamic_Viscosity": 0.224, "Dielectric_Constant": 1.84, "Surface_Tension": 16.0, "Vapor_Pressure": 56.8, "Water_Solubility": 38, "Hansen_Dispersion": 14.5, "Hansen_Polar": 0.0, "Hansen_Hydrogen": 0.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "hexane": {"Boiling_Point": 69.0, "Density": 0.659, "Dynamic_Viscosity": 0.300, "Dielectric_Constant": 1.88, "Surface_Tension": 18.4, "Vapor_Pressure": 16.2, "Water_Solubility": 9.5, "Hansen_Dispersion": 14.9, "Hansen_Polar": 0.0, "Hansen_Hydrogen": 0.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "heptane": {"Boiling_Point": 98.4, "Density": 0.684, "Dynamic_Viscosity": 0.387, "Dielectric_Constant": 1.92, "Surface_Tension": 20.1, "Vapor_Pressure": 4.7, "Water_Solubility": 2.24, "Hansen_Dispersion": 15.3, "Hansen_Polar": 0.0, "Hansen_Hydrogen": 0.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "octane": {"Boiling_Point": 125.7, "Density": 0.703, "Dynamic_Viscosity": 0.508, "Dielectric_Constant": 1.95, "Surface_Tension": 21.6, "Vapor_Pressure": 1.47, "Water_Solubility": 0.66, "Hansen_Dispersion": 15.5, "Hansen_Polar": 0.0, "Hansen_Hydrogen": 0.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "nonane": {"Boiling_Point": 150.8, "Density": 0.718, "Dynamic_Viscosity": 0.665, "Dielectric_Constant": 1.97, "Surface_Tension": 22.8, "Vapor_Pressure": 0.47, "Water_Solubility": 0.22, "Hansen_Dispersion": 15.7, "Hansen_Polar": 0.0, "Hansen_Hydrogen": 0.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "decane": {"Boiling_Point": 174.1, "Density": 0.730, "Dynamic_Viscosity": 0.838, "Dielectric_Constant": 1.99, "Surface_Tension": 23.8, "Vapor_Pressure": 0.17, "Water_Solubility": 0.052, "Hansen_Dispersion": 15.7, "Hansen_Polar": 0.0, "Hansen_Hydrogen": 0.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "dodecane": {"Boiling_Point": 216.3, "Density": 0.750, "Dynamic_Viscosity": 1.383, "Dielectric_Constant": 2.01, "Surface_Tension": 25.4, "Vapor_Pressure": 0.018, "Water_Solubility": 0.0034, "Hansen_Dispersion": 16.0, "Hansen_Polar": 0.0, "Hansen_Hydrogen": 0.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "cyclopentane": {"Boiling_Point": 49.3, "Density": 0.751, "Dynamic_Viscosity": 0.413, "Dielectric_Constant": 1.97, "Surface_Tension": 22.6, "Vapor_Pressure": 34.5, "Water_Solubility": 156, "Hansen_Dispersion": 16.4, "Hansen_Polar": 0.0, "Hansen_Hydrogen": 0.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "cyclohexane": {"Boiling_Point": 80.7, "Density": 0.779, "Dynamic_Viscosity": 0.894, "Dielectric_Constant": 2.02, "Surface_Tension": 25.2, "Vapor_Pressure": 10.4, "Water_Solubility": 55, "Hansen_Dispersion": 16.8, "Hansen_Polar": 0.0, "Hansen_Hydrogen": 0.2, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "methylcyclohexane": {"Boiling_Point": 101.0, "Density": 0.770, "Dynamic_Viscosity": 0.679, "Dielectric_Constant": 2.02, "Surface_Tension": 23.3, "Vapor_Pressure": 4.9, "Water_Solubility": 14, "Hansen_Dispersion": 16.0, "Hansen_Polar": 0.0, "Hansen_Hydrogen": 0.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "isooctane": {"Boiling_Point": 99.0, "Density": 0.692, "Dynamic_Viscosity": 0.474, "Dielectric_Constant": 1.94, "Surface_Tension": 18.8, "Vapor_Pressure": 5.13, "Water_Solubility": 2.44, "Hansen_Dispersion": 14.3, "Hansen_Polar": 0.0, "Hansen_Hydrogen": 0.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},

    # ── Aromatic Hydrocarbons ──
    "benzene": {"Boiling_Point": 80.1, "Density": 0.879, "Dynamic_Viscosity": 0.603, "Dielectric_Constant": 2.28, "Surface_Tension": 28.9, "Vapor_Pressure": 10.0, "Water_Solubility": 1790, "Hansen_Dispersion": 18.4, "Hansen_Polar": 0.0, "Hansen_Hydrogen": 2.0, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.0},
    "toluene": {"Boiling_Point": 110.6, "Density": 0.867, "Dynamic_Viscosity": 0.560, "Dielectric_Constant": 2.38, "Surface_Tension": 28.5, "Vapor_Pressure": 2.9, "Water_Solubility": 526, "Hansen_Dispersion": 18.0, "Hansen_Polar": 1.4, "Hansen_Hydrogen": 2.0, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.143},
    "o-xylene": {"Boiling_Point": 144.4, "Density": 0.880, "Dynamic_Viscosity": 0.760, "Dielectric_Constant": 2.57, "Surface_Tension": 30.1, "Vapor_Pressure": 0.88, "Water_Solubility": 178, "Hansen_Dispersion": 17.8, "Hansen_Polar": 1.0, "Hansen_Hydrogen": 3.1, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.25},
    "m-xylene": {"Boiling_Point": 139.1, "Density": 0.864, "Dynamic_Viscosity": 0.581, "Dielectric_Constant": 2.37, "Surface_Tension": 28.9, "Vapor_Pressure": 1.1, "Water_Solubility": 161, "Hansen_Dispersion": 17.4, "Hansen_Polar": 1.0, "Hansen_Hydrogen": 3.1, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.25},
    "p-xylene": {"Boiling_Point": 138.4, "Density": 0.861, "Dynamic_Viscosity": 0.603, "Dielectric_Constant": 2.27, "Surface_Tension": 28.3, "Vapor_Pressure": 1.17, "Water_Solubility": 162, "Hansen_Dispersion": 17.1, "Hansen_Polar": 1.0, "Hansen_Hydrogen": 3.1, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.25},
    "ethylbenzene": {"Boiling_Point": 136.2, "Density": 0.867, "Dynamic_Viscosity": 0.631, "Dielectric_Constant": 2.41, "Surface_Tension": 29.2, "Vapor_Pressure": 1.28, "Water_Solubility": 169, "Hansen_Dispersion": 17.8, "Hansen_Polar": 0.6, "Hansen_Hydrogen": 1.4, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.25},
    "cumene": {"Boiling_Point": 152.4, "Density": 0.862, "Dynamic_Viscosity": 0.737, "Dielectric_Constant": 2.38, "Surface_Tension": 28.2, "Vapor_Pressure": 0.61, "Water_Solubility": 61, "Hansen_Dispersion": 18.1, "Hansen_Polar": 1.2, "Hansen_Hydrogen": 1.2, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.333},
    "mesitylene": {"Boiling_Point": 164.7, "Density": 0.865, "Dynamic_Viscosity": 0.666, "Dielectric_Constant": 2.27, "Surface_Tension": 28.8, "Vapor_Pressure": 0.33, "Water_Solubility": 48, "Hansen_Dispersion": 18.0, "Hansen_Polar": 0.6, "Hansen_Hydrogen": 0.6, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.333},
    "styrene": {"Boiling_Point": 145.0, "Density": 0.909, "Dynamic_Viscosity": 0.695, "Dielectric_Constant": 2.43, "Surface_Tension": 32.0, "Vapor_Pressure": 0.67, "Water_Solubility": 300, "Hansen_Dispersion": 18.6, "Hansen_Polar": 1.0, "Hansen_Hydrogen": 4.1, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.0},
    "chlorobenzene": {"Boiling_Point": 131.7, "Density": 1.106, "Dynamic_Viscosity": 0.756, "Dielectric_Constant": 5.62, "Surface_Tension": 33.6, "Vapor_Pressure": 1.6, "Water_Solubility": 498, "Hansen_Dispersion": 19.0, "Hansen_Polar": 4.3, "Hansen_Hydrogen": 2.0, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.0},
    "o-dichlorobenzene": {"Boiling_Point": 180.5, "Density": 1.306, "Dynamic_Viscosity": 1.324, "Dielectric_Constant": 9.93, "Surface_Tension": 37.0, "Vapor_Pressure": 0.18, "Water_Solubility": 145, "Hansen_Dispersion": 19.2, "Hansen_Polar": 6.3, "Hansen_Hydrogen": 3.3, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.0},

    # ── Halogenated Hydrocarbons ──
    "dichloromethane": {"Boiling_Point": 39.6, "Density": 1.327, "Dynamic_Viscosity": 0.413, "Dielectric_Constant": 8.93, "Surface_Tension": 26.5, "Vapor_Pressure": 46.5, "Water_Solubility": 17500, "Hansen_Dispersion": 18.2, "Hansen_Polar": 6.3, "Hansen_Hydrogen": 6.1, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "chloroform": {"Boiling_Point": 61.2, "Density": 1.489, "Dynamic_Viscosity": 0.537, "Dielectric_Constant": 4.81, "Surface_Tension": 27.5, "Vapor_Pressure": 21.1, "Water_Solubility": 7950, "Hansen_Dispersion": 17.8, "Hansen_Polar": 3.1, "Hansen_Hydrogen": 5.7, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "carbon tetrachloride": {"Boiling_Point": 76.7, "Density": 1.594, "Dynamic_Viscosity": 0.908, "Dielectric_Constant": 2.24, "Surface_Tension": 27.0, "Vapor_Pressure": 11.9, "Water_Solubility": 793, "Hansen_Dispersion": 17.8, "Hansen_Polar": 0.0, "Hansen_Hydrogen": 0.6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "1,2-dichloroethane": {"Boiling_Point": 83.5, "Density": 1.253, "Dynamic_Viscosity": 0.779, "Dielectric_Constant": 10.4, "Surface_Tension": 31.9, "Vapor_Pressure": 8.0, "Water_Solubility": 8690, "Hansen_Dispersion": 19.0, "Hansen_Polar": 7.4, "Hansen_Hydrogen": 4.1, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "1,1,1-trichloroethane": {"Boiling_Point": 74.1, "Density": 1.339, "Dynamic_Viscosity": 0.793, "Dielectric_Constant": 7.5, "Surface_Tension": 25.6, "Vapor_Pressure": 13.3, "Water_Solubility": 4400, "Hansen_Dispersion": 17.0, "Hansen_Polar": 4.3, "Hansen_Hydrogen": 2.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "trichloroethylene": {"Boiling_Point": 87.2, "Density": 1.462, "Dynamic_Viscosity": 0.545, "Dielectric_Constant": 3.42, "Surface_Tension": 29.4, "Vapor_Pressure": 7.76, "Water_Solubility": 1100, "Hansen_Dispersion": 18.0, "Hansen_Polar": 3.1, "Hansen_Hydrogen": 5.3, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.0},
    "tetrachloroethylene": {"Boiling_Point": 121.3, "Density": 1.623, "Dynamic_Viscosity": 0.844, "Dielectric_Constant": 2.28, "Surface_Tension": 31.7, "Vapor_Pressure": 1.87, "Water_Solubility": 150, "Hansen_Dispersion": 19.0, "Hansen_Polar": 6.5, "Hansen_Hydrogen": 2.9, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.0},
    "1-chlorobutane": {"Boiling_Point": 78.4, "Density": 0.886, "Dynamic_Viscosity": 0.422, "Dielectric_Constant": 7.4, "Surface_Tension": 23.1, "Vapor_Pressure": 10.5, "Water_Solubility": 1100, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "fluorobenzene": {"Boiling_Point": 85.0, "Density": 1.024, "Dynamic_Viscosity": 0.550, "Dielectric_Constant": 5.42, "Surface_Tension": 26.7, "Vapor_Pressure": 7.0, "Water_Solubility": 1540, "Hansen_Dispersion": 17.7, "Hansen_Polar": 5.4, "Hansen_Hydrogen": 2.0, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.0},
    "trifluorotoluene": {"Boiling_Point": 103.0, "Density": 1.189, "Dynamic_Viscosity": 0.55, "Dielectric_Constant": 9.2, "Surface_Tension": 24.4, "Vapor_Pressure": 3.93, "Water_Solubility": 450, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.143},

    # ── Amines & Amides ──
    "methylamine": {"Boiling_Point": -6.3, "Density": 0.656, "Dynamic_Viscosity": 0.23, "Dielectric_Constant": 9.4, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "ethylamine": {"Boiling_Point": 16.6, "Density": 0.689, "Dynamic_Viscosity": 0.28, "Dielectric_Constant": 8.7, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "propylamine": {"Boiling_Point": 49.0, "Density": 0.719, "Dynamic_Viscosity": 0.35, "Dielectric_Constant": 5.1, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "butylamine": {"Boiling_Point": 77.8, "Density": 0.740, "Dynamic_Viscosity": 0.574, "Dielectric_Constant": 4.9, "Surface_Tension": 23.4, "Vapor_Pressure": 9.3, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "diethylamine": {"Boiling_Point": 55.5, "Density": 0.707, "Dynamic_Viscosity": 0.346, "Dielectric_Constant": 3.6, "Surface_Tension": 19.8, "Vapor_Pressure": 25.9, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "triethylamine": {"Boiling_Point": 88.8, "Density": 0.726, "Dynamic_Viscosity": 0.347, "Dielectric_Constant": 2.42, "Surface_Tension": 20.2, "Vapor_Pressure": 7.0, "Water_Solubility": 133000, "Hansen_Dispersion": 15.5, "Hansen_Polar": 0.4, "Hansen_Hydrogen": 1.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "diisopropylamine": {"Boiling_Point": 84.0, "Density": 0.722, "Dynamic_Viscosity": 0.393, "Dielectric_Constant": 2.9, "Surface_Tension": 19.6, "Vapor_Pressure": 7.3, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "aniline": {"Boiling_Point": 184.1, "Density": 1.022, "Dynamic_Viscosity": 3.71, "Dielectric_Constant": 6.89, "Surface_Tension": 43.4, "Vapor_Pressure": 0.04, "Water_Solubility": 36000, "Hansen_Dispersion": 19.4, "Hansen_Polar": 5.1, "Hansen_Hydrogen": 10.2, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.0},
    "pyridine": {"Boiling_Point": 115.2, "Density": 0.982, "Dynamic_Viscosity": 0.879, "Dielectric_Constant": 12.4, "Surface_Tension": 38.0, "Vapor_Pressure": 2.0, "Water_Solubility": 1e6, "Hansen_Dispersion": 19.0, "Hansen_Polar": 8.8, "Hansen_Hydrogen": 5.9, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.0},
    "piperidine": {"Boiling_Point": 106.0, "Density": 0.862, "Dynamic_Viscosity": 1.36, "Dielectric_Constant": 4.3, "Surface_Tension": 29.5, "Vapor_Pressure": 3.2, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "pyrrolidine": {"Boiling_Point": 87.0, "Density": 0.866, "Dynamic_Viscosity": 0.704, "Dielectric_Constant": 8.0, "Surface_Tension": 28.6, "Vapor_Pressure": 8.4, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "morpholine": {"Boiling_Point": 128.9, "Density": 1.000, "Dynamic_Viscosity": 2.02, "Dielectric_Constant": 7.42, "Surface_Tension": 38.0, "Vapor_Pressure": 1.07, "Water_Solubility": 1e6, "Hansen_Dispersion": 18.8, "Hansen_Polar": 4.9, "Hansen_Hydrogen": 9.2, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "formamide": {"Boiling_Point": 210.0, "Density": 1.133, "Dynamic_Viscosity": 3.343, "Dielectric_Constant": 111.0, "Surface_Tension": 58.2, "Vapor_Pressure": 0.01, "Water_Solubility": 1e6, "Hansen_Dispersion": 17.2, "Hansen_Polar": 26.2, "Hansen_Hydrogen": 19.0, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.0},
    "n,n-dimethylformamide": {"Boiling_Point": 153.0, "Density": 0.944, "Dynamic_Viscosity": 0.802, "Dielectric_Constant": 36.7, "Surface_Tension": 37.1, "Vapor_Pressure": 0.36, "Water_Solubility": 1e6, "Hansen_Dispersion": 17.4, "Hansen_Polar": 13.7, "Hansen_Hydrogen": 11.3, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.667},
    "n,n-dimethylacetamide": {"Boiling_Point": 165.0, "Density": 0.937, "Dynamic_Viscosity": 0.927, "Dielectric_Constant": 37.8, "Surface_Tension": 33.3, "Vapor_Pressure": 0.173, "Water_Solubility": 1e6, "Hansen_Dispersion": 16.8, "Hansen_Polar": 11.5, "Hansen_Hydrogen": 10.2, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.75},
    "n-methyl-2-pyrrolidone": {"Boiling_Point": 202.0, "Density": 1.028, "Dynamic_Viscosity": 1.666, "Dielectric_Constant": 32.2, "Surface_Tension": 40.7, "Vapor_Pressure": 0.039, "Water_Solubility": 1e6, "Hansen_Dispersion": 18.0, "Hansen_Polar": 12.3, "Hansen_Hydrogen": 7.2, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.6},

    # ── Nitriles ──
    "acetonitrile": {"Boiling_Point": 82.0, "Density": 0.786, "Dynamic_Viscosity": 0.369, "Dielectric_Constant": 37.5, "Surface_Tension": 29.3, "Vapor_Pressure": 9.7, "Water_Solubility": 1e6, "Hansen_Dispersion": 15.3, "Hansen_Polar": 18.0, "Hansen_Hydrogen": 6.1, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.5},
    "propionitrile": {"Boiling_Point": 97.4, "Density": 0.782, "Dynamic_Viscosity": 0.404, "Dielectric_Constant": 28.9, "Surface_Tension": 26.8, "Vapor_Pressure": 4.27, "Water_Solubility": 103000, "Hansen_Dispersion": 15.3, "Hansen_Polar": 14.3, "Hansen_Hydrogen": 5.5, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.667},
    "butyronitrile": {"Boiling_Point": 117.6, "Density": 0.794, "Dynamic_Viscosity": 0.553, "Dielectric_Constant": 24.3, "Surface_Tension": 28.1, "Vapor_Pressure": 1.65, "Water_Solubility": 33000, "Hansen_Dispersion": 15.3, "Hansen_Polar": 12.4, "Hansen_Hydrogen": 5.1, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.75},
    "benzonitrile": {"Boiling_Point": 191.0, "Density": 1.010, "Dynamic_Viscosity": 1.267, "Dielectric_Constant": 25.2, "Surface_Tension": 39.0, "Vapor_Pressure": 0.12, "Water_Solubility": 2000, "Hansen_Dispersion": 17.4, "Hansen_Polar": 9.0, "Hansen_Hydrogen": 3.3, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.0},

    # ── Sulfoxides & Sulfones ──
    "dimethyl sulfoxide": {"Boiling_Point": 189.0, "Density": 1.100, "Dynamic_Viscosity": 1.996, "Dielectric_Constant": 46.7, "Surface_Tension": 43.5, "Vapor_Pressure": 0.06, "Water_Solubility": 1e6, "Hansen_Dispersion": 18.4, "Hansen_Polar": 16.4, "Hansen_Hydrogen": 10.2, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "sulfolane": {"Boiling_Point": 287.3, "Density": 1.261, "Dynamic_Viscosity": 10.3, "Dielectric_Constant": 43.3, "Surface_Tension": 35.5, "Vapor_Pressure": 0.001, "Water_Solubility": 1e6, "Hansen_Dispersion": 20.3, "Hansen_Polar": 18.2, "Hansen_Hydrogen": 9.9, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},

    # ── Carboxylic Acids ──
    "formic acid": {"Boiling_Point": 100.8, "Density": 1.220, "Dynamic_Viscosity": 1.607, "Dielectric_Constant": 58.5, "Surface_Tension": 37.6, "Vapor_Pressure": 4.4, "Water_Solubility": 1e6, "Hansen_Dispersion": 14.3, "Hansen_Polar": 11.9, "Hansen_Hydrogen": 16.6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.0},
    "acetic acid": {"Boiling_Point": 117.9, "Density": 1.049, "Dynamic_Viscosity": 1.056, "Dielectric_Constant": 6.15, "Surface_Tension": 27.1, "Vapor_Pressure": 1.5, "Water_Solubility": 1e6, "Hansen_Dispersion": 14.5, "Hansen_Polar": 8.0, "Hansen_Hydrogen": 13.5, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.5},
    "propionic acid": {"Boiling_Point": 141.2, "Density": 0.990, "Dynamic_Viscosity": 1.03, "Dielectric_Constant": 3.4, "Surface_Tension": 26.2, "Vapor_Pressure": 0.47, "Water_Solubility": 1e6, "Hansen_Dispersion": 14.7, "Hansen_Polar": 5.3, "Hansen_Hydrogen": 12.4, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.667},
    "butyric acid": {"Boiling_Point": 163.8, "Density": 0.964, "Dynamic_Viscosity": 1.54, "Dielectric_Constant": 2.97, "Surface_Tension": 26.5, "Vapor_Pressure": 0.1, "Water_Solubility": 1e6, "Hansen_Dispersion": 14.9, "Hansen_Polar": 4.1, "Hansen_Hydrogen": 10.6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.75},
    "valeric acid": {"Boiling_Point": 186.1, "Density": 0.939, "Dynamic_Viscosity": 2.01, "Dielectric_Constant": 2.66, "Surface_Tension": 27.0, "Vapor_Pressure": 0.03, "Water_Solubility": 24000, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.8},
    "hexanoic acid": {"Boiling_Point": 205.0, "Density": 0.929, "Dynamic_Viscosity": 2.82, "Dielectric_Constant": 2.63, "Surface_Tension": 27.6, "Vapor_Pressure": 0.011, "Water_Solubility": 9680, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.833},
    "trifluoroacetic acid": {"Boiling_Point": 72.4, "Density": 1.489, "Dynamic_Viscosity": 0.808, "Dielectric_Constant": 8.55, "Surface_Tension": 13.6, "Vapor_Pressure": 11.1, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.5},

    # ── Nitro Compounds ──
    "nitromethane": {"Boiling_Point": 101.2, "Density": 1.138, "Dynamic_Viscosity": 0.630, "Dielectric_Constant": 35.9, "Surface_Tension": 36.5, "Vapor_Pressure": 3.61, "Water_Solubility": 111000, "Hansen_Dispersion": 15.8, "Hansen_Polar": 18.8, "Hansen_Hydrogen": 5.1, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
    "nitroethane": {"Boiling_Point": 114.0, "Density": 1.052, "Dynamic_Viscosity": 0.677, "Dielectric_Constant": 28.1, "Surface_Tension": 32.2, "Vapor_Pressure": 2.0, "Water_Solubility": 45000, "Hansen_Dispersion": 16.0, "Hansen_Polar": 15.5, "Hansen_Hydrogen": 4.5, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.5},
    "nitrobenzene": {"Boiling_Point": 210.8, "Density": 1.204, "Dynamic_Viscosity": 1.863, "Dielectric_Constant": 34.8, "Surface_Tension": 43.9, "Vapor_Pressure": 0.025, "Water_Solubility": 1900, "Hansen_Dispersion": 20.0, "Hansen_Polar": 8.6, "Hansen_Hydrogen": 4.1, "Aromatic_Ring_Count": 1, "Fraction_CSP3": 0.0},

    # ── Miscellaneous ──
    "carbon disulfide": {"Boiling_Point": 46.3, "Density": 1.263, "Dynamic_Viscosity": 0.352, "Dielectric_Constant": 2.63, "Surface_Tension": 32.3, "Vapor_Pressure": 35.8, "Water_Solubility": 2160, "Hansen_Dispersion": 20.5, "Hansen_Polar": 0.0, "Hansen_Hydrogen": 0.6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 0.0},
    "hexamethylphosphoramide": {"Boiling_Point": 233.0, "Density": 1.030, "Dynamic_Viscosity": 3.47, "Dielectric_Constant": 30.0, "Surface_Tension": 34.0, "Vapor_Pressure": 0.007, "Water_Solubility": 1e6, "Aromatic_Ring_Count": 0, "Fraction_CSP3": 1.0},
}
# fmt: on

# ---------------------------------------------------------------------------
# PubChem helper – batch fetch HeavyAtomCount by CID
# ---------------------------------------------------------------------------

def fetch_heavy_atom_counts(cids: list[int]) -> dict[int, int]:
    """Fetch HeavyAtomCount from PubChem for a list of CIDs."""
    result = {}
    batch_size = 100
    for i in range(0, len(cids), batch_size):
        batch = cids[i : i + batch_size]
        cid_str = ",".join(str(int(c)) for c in batch)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/HeavyAtomCount/JSON"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                for prop in data.get("PropertyTable", {}).get("Properties", []):
                    result[prop["CID"]] = prop.get("HeavyAtomCount")
                log.info(f"  Fetched HeavyAtomCount for CIDs batch {i+1}-{i+len(batch)}")
            else:
                log.warning(f"  PubChem returned {resp.status_code} for batch {i+1}")
        except Exception as e:
            log.warning(f"  Error fetching HeavyAtomCount batch: {e}")
        time.sleep(0.5)
    return result


def main():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "universal_solvent_database.csv")
    if not os.path.exists(csv_path):
        log.error(f"Database not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    log.info(f"Loaded database: {df.shape[0]} solvents x {df.shape[1]} columns")

    # ── Step 1: Fetch HeavyAtomCount from PubChem ──
    if "PubChem_CID" in df.columns:
        valid_cids = df.loc[df["PubChem_CID"].notna(), "PubChem_CID"].astype(int).tolist()
        if valid_cids:
            log.info(f"Fetching HeavyAtomCount from PubChem for {len(valid_cids)} CIDs...")
            hac_map = fetch_heavy_atom_counts(valid_cids)
            filled = 0
            for idx, row in df.iterrows():
                cid = row.get("PubChem_CID")
                if pd.notna(cid) and int(cid) in hac_map:
                    val = hac_map[int(cid)]
                    if val is not None and (pd.isna(row.get("Heavy_Atom_Count")) or row.get("Heavy_Atom_Count") == 0):
                        df.at[idx, "Heavy_Atom_Count"] = val
                        filled += 1
            log.info(f"  Filled Heavy_Atom_Count for {filled} solvents from PubChem.")

    # ── Step 2: Apply curated literature data ──
    log.info("Applying curated literature data...")
    filled_counts = {}
    for idx, row in df.iterrows():
        name_lower = str(row["Solvent_Name"]).strip().lower()
        lit = LITERATURE.get(name_lower)
        if not lit:
            continue
        for col, val in lit.items():
            if col in df.columns and (pd.isna(row.get(col)) or row.get(col) == 0):
                df.at[idx, col] = val
                filled_counts[col] = filled_counts.get(col, 0) + 1

    log.info("  Literature fill summary:")
    for col in sorted(filled_counts.keys()):
        log.info(f"    {col}: filled {filled_counts[col]} values")

    # ── Step 3: Save ──
    df.to_csv(csv_path, index=False)
    log.info(f"\nSaved enriched database to {csv_path}")

    # ── Summary ──
    log.info("\n=== Database Completeness ===")
    for col in df.columns:
        n_filled = df[col].notna().sum()
        n_total = len(df)
        pct = 100 * n_filled / n_total
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        log.info(f"  {col:35s}  {bar}  {n_filled:3d}/{n_total}  ({pct:.0f}%)")


if __name__ == "__main__":
    main()
