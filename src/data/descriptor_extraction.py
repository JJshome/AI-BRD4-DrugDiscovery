#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Molecular Descriptor Extraction Module

This module provides functionality to calculate molecular descriptors for ligands.
These descriptors represent the ligands' properties that can be used for machine learning models.
It implements the second step in the AI-based drug development method as described in the patent.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.EState import Fingerprinter as EStateFingerprinter
from rdkit.Chem import AtomPairs
from rdkit.Chem.Fingerprints import FingerprintMols
from mordred import Calculator, descriptors
import logging
import os
from typing import List, Dict, Optional, Union, Tuple
import pickle
from collections import defaultdict
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DescriptorExtractor:
    """Class for extracting molecular descriptors from ligands."""
    
    def __init__(self, use_mordred=True, use_rdkit=True, use_fingerprints=True):
        """
        Initialize the DescriptorExtractor object.
        
        Args:
            use_mordred (bool, optional): Whether to calculate Mordred descriptors. Defaults to True.
            use_rdkit (bool, optional): Whether to calculate RDKit descriptors. Defaults to True.
            use_fingerprints (bool, optional): Whether to calculate fingerprints. Defaults to True.
        """
        self.use_mordred = use_mordred
        self.use_rdkit = use_rdkit
        self.use_fingerprints = use_fingerprints
        
        # Initialize calculators
        if use_mordred:
            self.mordred_calc = Calculator(descriptors, ignore_3D=False)
        
        if use_rdkit:
            # Get all RDKit descriptor names
            self.rdkit_desc_names = [x[0] for x in Descriptors._descList]
            self.rdkit_calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.rdkit_desc_names)
    
    def _calc_mordred_descriptors(self, mol):
        """
        Calculate Mordred descriptors for a molecule.
        
        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.
            
        Returns:
            dict: Dictionary of Mordred descriptors.
        """
        try:
            result = self.mordred_calc(mol)
            # Convert to dictionary
            result_dict = {str(k): float(v) if v is not None else np.nan for k, v in result.items()}
            return result_dict
        except Exception as e:
            logger.warning(f"Error calculating Mordred descriptors: {e}")
            return {}
    
    def _calc_rdkit_descriptors(self, mol):
        """
        Calculate RDKit descriptors for a molecule.
        
        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.
            
        Returns:
            dict: Dictionary of RDKit descriptors.
        """
        try:
            desc_values = self.rdkit_calc.CalcDescriptors(mol)
            return {f"rdkit_{name}": value for name, value in zip(self.rdkit_desc_names, desc_values)}
        except Exception as e:
            logger.warning(f"Error calculating RDKit descriptors: {e}")
            return {}
    
    def _calc_fingerprints(self, mol):
        """
        Calculate various molecular fingerprints for a molecule.
        
        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.
            
        Returns:
            dict: Dictionary of fingerprint descriptors.
        """
        result = {}
        
        try:
            # Morgan Fingerprints (ECFP)
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            morgan_bits = {f"Morgan_{i}": int(morgan_fp[i]) for i in range(len(morgan_fp))}
            result.update(morgan_bits)
            
            # MACCS Keys
            maccs = MACCSkeys.GenMACCSKeys(mol)
            maccs_bits = {f"MACCS_{i}": int(maccs[i]) for i in range(len(maccs))}
            result.update(maccs_bits)
            
            # Topological Torsion Fingerprints
            tt_fp = AtomPairs.GetTopologicalTorsionFingerprint(mol)
            tt_bits = {f"TT_{i[0]}": i[1] for i in tt_fp.GetNonzeroElements().items()}
            result.update(tt_bits)
            
            # Atom Pair Fingerprints
            ap_fp = AtomPairs.GetAtomPairFingerprint(mol)
            ap_bits = {f"AP_{i[0]}": i[1] for i in ap_fp.GetNonzeroElements().items()}
            result.update(ap_bits)
        except Exception as e:
            logger.warning(f"Error calculating fingerprints: {e}")
        
        return result
    
    def calculate_descriptors(self, smiles_list, njobs=1, chunk_size=100):
        """
        Calculate molecular descriptors for a list of SMILES strings.
        
        Args:
            smiles_list (list): List of SMILES strings.
            njobs (int, optional): Number of parallel jobs. Defaults to 1.
            chunk_size (int, optional): Chunk size for processing. Defaults to 100.
            
        Returns:
            pandas.DataFrame: DataFrame containing calculated descriptors.
        """
        start_time = time.time()
        logger.info(f"Calculating descriptors for {len(smiles_list)} molecules...")
        
        # Convert SMILES to RDKit molecules
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        
        # Filter out None values (invalid SMILES)
        valid_mols = [(i, mol) for i, mol in enumerate(mols) if mol is not None]
        invalid_indices = [i for i, mol in enumerate(mols) if mol is None]
        
        if invalid_indices:
            logger.warning(f"Found {len(invalid_indices)} invalid SMILES strings at indices: {invalid_indices}")
        
        # Calculate descriptors
        all_descriptors = []
        
        # Process molecules in chunks to avoid memory issues
        for i in range(0, len(valid_mols), chunk_size):
            chunk = valid_mols[i:i+chunk_size]
            chunk_results = []
            
            for idx, mol in chunk:
                # Initialize with molecule index and SMILES
                desc = {"molecule_idx": idx, "SMILES": smiles_list[idx]}
                
                # Calculate various descriptors
                if self.use_mordred:
                    mordred_desc = self._calc_mordred_descriptors(mol)
                    desc.update(mordred_desc)
                
                if self.use_rdkit:
                    rdkit_desc = self._calc_rdkit_descriptors(mol)
                    desc.update(rdkit_desc)
                
                if self.use_fingerprints:
                    fp_desc = self._calc_fingerprints(mol)
                    desc.update(fp_desc)
                
                chunk_results.append(desc)
            
            all_descriptors.extend(chunk_results)
            logger.info(f"Processed {len(all_descriptors)}/{len(valid_mols)} molecules...")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_descriptors)
        
        # Handle missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Calculate completion time
        elapsed_time = time.time() - start_time
        logger.info(f"Descriptor calculation completed in {elapsed_time:.2f} seconds.")
        logger.info(f"Generated {df.shape[1] - 2} descriptors for {df.shape[0]} molecules.")
        
        return df
    
    def extract_from_dataframe(self, df, smiles_col="canonical_smiles", id_col=None, save_path=None):
        """
        Extract descriptors from a DataFrame containing SMILES.
        
        Args:
            df (pandas.DataFrame): DataFrame containing SMILES.
            smiles_col (str, optional): Column name for SMILES. Defaults to "canonical_smiles".
            id_col (str, optional): Column name for molecule IDs. Defaults to None.
            save_path (str, optional): Path to save the descriptors. Defaults to None.
            
        Returns:
            pandas.DataFrame: DataFrame containing calculated descriptors.
        """
        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found in DataFrame.")
        
        # Extract SMILES list
        smiles_list = df[smiles_col].tolist()
        
        # Calculate descriptors
        desc_df = self.calculate_descriptors(smiles_list)
        
        # Merge with original data if ID column is provided
        if id_col is not None:
            if id_col not in df.columns:
                logger.warning(f"ID column '{id_col}' not found in DataFrame. Skipping merge.")
            else:
                # Add original IDs to the descriptor DataFrame
                desc_df = desc_df.merge(df[[id_col, smiles_col]], left_on="SMILES", right_on=smiles_col)
        
        # Save to file if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            desc_df.to_csv(save_path, index=False)
            logger.info(f"Saved descriptors to {save_path}")
        
        return desc_df
    
    def filter_descriptors(self, desc_df, min_variance=0.01, correlation_threshold=0.95, 
                          remove_missing=True, missing_threshold=0.2):
        """
        Filter descriptors based on variance, correlation, and missing values.
        
        Args:
            desc_df (pandas.DataFrame): DataFrame containing descriptors.
            min_variance (float, optional): Minimum variance required. Defaults to 0.01.
            correlation_threshold (float, optional): Correlation threshold for removing highly correlated features.
                                                  Defaults to 0.95.
            remove_missing (bool, optional): Whether to remove features with missing values. Defaults to True.
            missing_threshold (float, optional): Threshold for maximum missing values ratio. Defaults to 0.2.
            
        Returns:
            pandas.DataFrame: Filtered DataFrame containing descriptors.
        """
        logger.info(f"Original descriptor shape: {desc_df.shape}")
        
        # Make a copy to avoid modifying the original
        filtered_df = desc_df.copy()
        
        # Get feature columns (excluding molecule_idx and SMILES)
        feature_cols = [col for col in filtered_df.columns if col not in ["molecule_idx", "SMILES", "LigandID", "canonical_smiles"]]
        
        # Remove features with too many missing values
        if remove_missing:
            missing_ratio = filtered_df[feature_cols].isnull().mean()
            cols_to_keep = missing_ratio[missing_ratio <= missing_threshold].index.tolist()
            removed_cols = set(feature_cols) - set(cols_to_keep)
            logger.info(f"Removed {len(removed_cols)} features due to missing values.")
            
            filtered_df = filtered_df[["molecule_idx", "SMILES"] + cols_to_keep]
            feature_cols = cols_to_keep
        
        # Remove features with low variance
        variance = filtered_df[feature_cols].var()
        high_var_cols = variance[variance > min_variance].index.tolist()
        removed_cols = set(feature_cols) - set(high_var_cols)
        logger.info(f"Removed {len(removed_cols)} features due to low variance.")
        
        filtered_df = filtered_df[["molecule_idx", "SMILES"] + high_var_cols]
        feature_cols = high_var_cols
        
        # Remove highly correlated features
        corr_matrix = filtered_df[feature_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        logger.info(f"Removed {len(to_drop)} features due to high correlation.")
        
        filtered_df = filtered_df.drop(columns=to_drop)
        
        logger.info(f"Final descriptor shape after filtering: {filtered_df.shape}")
        return filtered_df
    
    def save_descriptors(self, desc_df, path, include_metadata=True):
        """
        Save descriptors to a file.
        
        Args:
            desc_df (pandas.DataFrame): DataFrame containing descriptors.
            path (str): Path to save the descriptors.
            include_metadata (bool, optional): Whether to include metadata. Defaults to True.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save CSV
        desc_df.to_csv(path, index=False)
        
        # Save metadata if requested
        if include_metadata:
            metadata = {
                "num_molecules": desc_df.shape[0],
                "num_descriptors": desc_df.shape[1] - 2,  # Exclude molecule_idx and SMILES
                "descriptor_names": [col for col in desc_df.columns if col not in ["molecule_idx", "SMILES", "LigandID", "canonical_smiles"]],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            metadata_path = os.path.splitext(path)[0] + "_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved descriptor metadata to {metadata_path}")
        
        logger.info(f"Saved {desc_df.shape[0]} molecules with {desc_df.shape[1] - 2} descriptors to {path}")
    
    def load_descriptors(self, path):
        """
        Load descriptors from a file.
        
        Args:
            path (str): Path to load the descriptors from.
            
        Returns:
            pandas.DataFrame: Loaded descriptors.
        """
        if not os.path.exists(path):
            raise ValueError(f"File not found: {path}")
        
        desc_df = pd.read_csv(path)
        logger.info(f"Loaded {desc_df.shape[0]} molecules with {desc_df.shape[1] - 2} descriptors from {path}")
        return desc_df

# Example usage
if __name__ == "__main__":
    # Example usage with sample data
    from src.data.ligand_selection import LigandSelector
    
    # Load ligands
    selector = LigandSelector()
    ligands = selector.load_ligands("data/brd4_ligands.csv")
    
    # Extract descriptors
    extractor = DescriptorExtractor(use_mordred=True, use_rdkit=True, use_fingerprints=True)
    
    # Calculate descriptors
    desc_df = extractor.extract_from_dataframe(
        ligands, 
        smiles_col="canonical_smiles", 
        id_col="molecule_chembl_id",
        save_path="data/brd4_descriptors_raw.csv"
    )
    
    # Filter descriptors
    filtered_desc_df = extractor.filter_descriptors(
        desc_df,
        min_variance=0.01,
        correlation_threshold=0.95,
        remove_missing=True,
        missing_threshold=0.2
    )
    
    # Save filtered descriptors
    extractor.save_descriptors(filtered_desc_df, "data/brd4_descriptors_filtered.csv")
