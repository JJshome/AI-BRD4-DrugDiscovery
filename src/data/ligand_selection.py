#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ligand Selection Module for BRD4 Target Protein

This module provides functions to select appropriate ligands for a target protein (BRD4)
using ChEMBL database and other resources. It implements the first step in the AI-based
drug development method as described in the patent.
"""

import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import os
import logging
import json
from typing import List, Dict, Optional, Union, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LigandSelector:
    """Class for selecting ligands that bind to a target protein."""
    
    def __init__(self, target_name=None, target_chembl_id=None, assay_type="IC50", threshold=None):
        """
        Initialize the LigandSelector object.
        
        Args:
            target_name (str, optional): Name of the target protein (e.g., 'BRD4').
            target_chembl_id (str, optional): ChEMBL ID of the target protein (e.g., 'CHEMBL1163125' for BRD4).
            assay_type (str, optional): Type of assay to filter for (e.g., 'IC50', 'Ki', 'Kd'). Defaults to 'IC50'.
            threshold (float, optional): Value threshold for filtering ligands. Defaults to None.
        """
        self.target_name = target_name
        self.target_chembl_id = target_chembl_id
        self.assay_type = assay_type
        self.threshold = threshold
        self.ligands = None
        self.client = new_client()
        
    def search_target(self, query=None):
        """
        Search for a target in ChEMBL database.
        
        Args:
            query (str, optional): Query string to search for targets. If None, uses self.target_name.
            
        Returns:
            list: List of target dictionaries.
        """
        if query is None and self.target_name is None:
            raise ValueError("Either query or target_name must be provided.")
        
        search_query = query if query is not None else self.target_name
        logger.info(f"Searching for target: {search_query}")
        
        target = self.client.target
        target_query = target.filter(pref_name__icontains=search_query)
        targets = target_query.only(['target_chembl_id', 'pref_name', 'target_type', 'organism'])
        
        return targets
    
    def set_target(self, target_chembl_id):
        """
        Set the target ChEMBL ID for the selector.
        
        Args:
            target_chembl_id (str): ChEMBL ID of the target.
        """
        self.target_chembl_id = target_chembl_id
        logger.info(f"Target set to: {target_chembl_id}")
    
    def fetch_ligands(self, limit=None, save_path=None):
        """
        Fetch ligands for the target from ChEMBL database.
        
        Args:
            limit (int, optional): Maximum number of ligands to fetch. None means no limit.
            save_path (str, optional): Path to save the fetched ligands. None means don't save.
            
        Returns:
            pandas.DataFrame: DataFrame containing ligand information.
        """
        if self.target_chembl_id is None:
            raise ValueError("Target ChEMBL ID must be set before fetching ligands.")
        
        logger.info(f"Fetching ligands for target: {self.target_chembl_id}, assay type: {self.assay_type}")
        
        activities = self.client.activity
        query = activities.filter(
            target_chembl_id=self.target_chembl_id,
            standard_type=self.assay_type
        ).only(['molecule_chembl_id', 'standard_value', 'standard_units', 'standard_type', 
                'activity_id', 'assay_chembl_id', 'canonical_smiles'])
        
        if limit:
            query = query[:limit]
        
        # Convert to DataFrame
        ligands_df = pd.DataFrame(query)
        
        # Filter out rows with missing SMILES
        ligands_df = ligands_df.dropna(subset=['canonical_smiles'])
        
        # Apply threshold filter if provided
        if self.threshold is not None:
            ligands_df = ligands_df[ligands_df['standard_value'] <= self.threshold]
        
        # Store the ligands
        self.ligands = ligands_df
        
        # Save to file if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            ligands_df.to_csv(save_path, index=False)
            logger.info(f"Saved {len(ligands_df)} ligands to {save_path}")
        
        logger.info(f"Fetched {len(ligands_df)} ligands.")
        return ligands_df
    
    def filter_ligands(self, property_filters=None, similarity_filters=None):
        """
        Filter ligands based on property and similarity criteria.
        
        Args:
            property_filters (dict, optional): Dictionary of property filters, e.g., {'molecular_weight': (0, 500)}.
            similarity_filters (dict, optional): Dictionary of similarity filters, e.g., 
                                              {'reference_smiles': 'CCO', 'threshold': 0.7}.
                                              
        Returns:
            pandas.DataFrame: Filtered DataFrame of ligands.
        """
        if self.ligands is None:
            raise ValueError("No ligands to filter. Call fetch_ligands first.")
        
        filtered_df = self.ligands.copy()
        
        # Apply property filters
        if property_filters:
            for prop, (min_val, max_val) in property_filters.items():
                if prop == 'molecular_weight':
                    filtered_df['mol'] = filtered_df['canonical_smiles'].apply(Chem.MolFromSmiles)
                    filtered_df['molecular_weight'] = filtered_df['mol'].apply(
                        lambda x: Descriptors.MolWt(x) if x else None
                    )
                    filtered_df = filtered_df[
                        (filtered_df['molecular_weight'] >= min_val) & 
                        (filtered_df['molecular_weight'] <= max_val)
                    ]
                    filtered_df = filtered_df.drop(['mol', 'molecular_weight'], axis=1)
                # Add other property filters as needed
        
        # Apply similarity filters
        if similarity_filters and 'reference_smiles' in similarity_filters:
            ref_mol = Chem.MolFromSmiles(similarity_filters['reference_smiles'])
            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
            
            threshold = similarity_filters.get('threshold', 0.7)
            
            # Create molecules from SMILES
            filtered_df['mol'] = filtered_df['canonical_smiles'].apply(Chem.MolFromSmiles)
            
            # Calculate similarity
            def calc_similarity(mol):
                if mol is None:
                    return 0
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                return DataStructs.TanimotoSimilarity(ref_fp, fp)
            
            filtered_df['similarity'] = filtered_df['mol'].apply(calc_similarity)
            filtered_df = filtered_df[filtered_df['similarity'] >= threshold]
            
            # Optional: keep the similarity score
            # filtered_df = filtered_df.drop(['mol'], axis=1)
        
        logger.info(f"Filtered to {len(filtered_df)} ligands.")
        return filtered_df
    
    def select_diverse_subset(self, n_ligands=100, method='MaxMin'):
        """
        Select a diverse subset of ligands using RDKit's diversity picking.
        
        Args:
            n_ligands (int, optional): Number of ligands to select. Defaults to 100.
            method (str, optional): Method for diversity selection. Defaults to 'MaxMin'.
            
        Returns:
            pandas.DataFrame: Selected diverse ligands.
        """
        if self.ligands is None or len(self.ligands) == 0:
            raise ValueError("No ligands to select from. Call fetch_ligands first.")
        
        if len(self.ligands) <= n_ligands:
            logger.info(f"Requested {n_ligands} ligands, but only {len(self.ligands)} available. Returning all.")
            return self.ligands
        
        # Create molecules from SMILES
        mols = [Chem.MolFromSmiles(smi) for smi in self.ligands['canonical_smiles']]
        valid_mols_idx = [i for i, m in enumerate(mols) if m is not None]
        valid_mols = [mols[i] for i in valid_mols_idx]
        
        # Calculate fingerprints
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in valid_mols]
        
        # Select diverse subset
        from rdkit.SimDivFilters import rdSimDivPickers
        if method == 'MaxMin':
            picker = rdSimDivPickers.MaxMinPicker()
            pickIndices = picker.LazyPick(fps, len(fps), n_ligands)
        else:
            # Default to MaxMin
            picker = rdSimDivPickers.MaxMinPicker()
            pickIndices = picker.LazyPick(fps, len(fps), n_ligands)
        
        # Map back to original indices
        selected_indices = [valid_mols_idx[i] for i in pickIndices]
        selected_ligands = self.ligands.iloc[selected_indices].reset_index(drop=True)
        
        logger.info(f"Selected {len(selected_ligands)} diverse ligands.")
        return selected_ligands
    
    def save_ligands(self, ligands_df=None, path='data/selected_ligands.csv'):
        """
        Save ligands to a file.
        
        Args:
            ligands_df (pandas.DataFrame, optional): DataFrame of ligands to save. If None, uses self.ligands.
            path (str, optional): Path to save the ligands. Defaults to 'data/selected_ligands.csv'.
        """
        df_to_save = ligands_df if ligands_df is not None else self.ligands
        
        if df_to_save is None or len(df_to_save) == 0:
            raise ValueError("No ligands to save.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df_to_save.to_csv(path, index=False)
        logger.info(f"Saved {len(df_to_save)} ligands to {path}")
        
    def load_ligands(self, path):
        """
        Load ligands from a file.
        
        Args:
            path (str): Path to load the ligands from.
            
        Returns:
            pandas.DataFrame: Loaded ligands.
        """
        if not os.path.exists(path):
            raise ValueError(f"File not found: {path}")
        
        self.ligands = pd.read_csv(path)
        logger.info(f"Loaded {len(self.ligands)} ligands from {path}")
        return self.ligands

# Example usage
if __name__ == "__main__":
    # Example for BRD4
    selector = LigandSelector(target_name="BRD4")
    
    # Search for BRD4 in ChEMBL
    targets = selector.search_target()
    
    if targets:
        # Assuming the first result is what we want
        target_info = targets[0]
        print(f"Found target: {target_info['pref_name']} ({target_info['target_chembl_id']})")
        
        # Set the target and fetch ligands
        selector.set_target(target_info['target_chembl_id'])
        ligands = selector.fetch_ligands(limit=2000, save_path="data/brd4_ligands.csv")
        
        # Filter ligands (optional)
        filtered_ligands = selector.filter_ligands(
            property_filters={'molecular_weight': (200, 600)}
        )
        
        # Select a diverse subset
        diverse_ligands = selector.select_diverse_subset(n_ligands=100)
        
        # Save the selected ligands
        selector.save_ligands(diverse_ligands, path="data/brd4_diverse_ligands.csv")
    else:
        print("No targets found for BRD4.")
