#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive simulation for BRD4 inhibitor prediction

This script provides a Streamlit-based web interface for testing and visualizing
BRD4 inhibitor predictions using the trained models.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, PandasTools
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import project modules
from src.data.descriptor_extraction import extract_descriptors
from src.models.prediction import predict_activity


# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'brd4_model.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'brd4_scaler.pkl')
        features_path = os.path.join(os.path.dirname(__file__), 'models', 'required_features.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        with open(features_path, 'rb') as f:
            required_features = pickle.load(f)
            
        return model, scaler, required_features
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Provide demo model
        return None, None, None


# Process SMILES input
def process_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES string"
    
    # Create DataFrame with single molecule
    df = pd.DataFrame({'smiles': [smiles]})
    
    # Extract descriptors
    descriptors_df = extract_descriptors(df)
    
    return descriptors_df, None


# Process SDF file input
def process_sdf(sdf_file):
    try:
        supplier = Chem.SDMolSupplier(sdf_file)
        molecules = [mol for mol in supplier if mol is not None]
        
        if not molecules:
            return None, "No valid molecules found in SDF file"
        
        # Extract SMILES
        smiles_list = [Chem.MolToSmiles(mol) for mol in molecules]
        
        # Create DataFrame
        df = pd.DataFrame({'smiles': smiles_list})
        
        # Add molecule properties if available
        for i, mol in enumerate(molecules):
            for prop_name in mol.GetPropNames():
                if prop_name not in df.columns:
                    df[prop_name] = None
                df.loc[i, prop_name] = mol.GetProp(prop_name)
        
        # Extract descriptors
        descriptors_df = extract_descriptors(df)
        
        # Add original SMILES and properties
        for col in df.columns:
            if col != 'smiles':  # smiles is already used for descriptor calculation
                descriptors_df[col] = df[col].values
        
        return descriptors_df, None
    except Exception as e:
        return None, f"Error processing SDF file: {e}"


# Main application
def main():
    st.title("BRD4 Inhibitor Prediction Simulation")
    
    st.markdown("""
    This simulation tool allows you to predict the inhibitory activity (IC50) of compounds against BRD4 protein.
    You can input compounds using SMILES strings or upload an SDF file containing multiple compounds.
    """)
    
    # Load model
    model, scaler, required_features = load_model()
    
    if model is None:
        st.warning("Running in demo mode with a simulated model. Predictions will be random.")
    
    # Sidebar
    st.sidebar.header("Input Options")
    input_method = st.sidebar.radio("Select input method", ["SMILES", "SDF File"])
    activity_threshold = st.sidebar.slider("Activity threshold (IC50 in nM)", 1, 10000, 1000, 100)
    
    # Main area
    if input_method == "SMILES":
        smiles_input = st.text_area("Enter SMILES string(s)", "CC(=O)Nc1ccc(O)cc1")
        process_button = st.button("Process SMILES")
        
        if process_button:
            if smiles_input.strip():
                # Split multiple SMILES if provided
                smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
                
                all_results = []
                all_molecules = []
                
                for smiles in smiles_list:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        st.error(f"Invalid SMILES: {smiles}")
                        continue
                        
                    all_molecules.append(mol)
                    
                    # Process single SMILES
                    descriptors_df, error = process_smiles(smiles)
                    
                    if error:
                        st.error(error)
                        continue
                    
                    # Make prediction
                    if model is not None:
                        result = predict_activity(model, descriptors_df, scaler, required_features, 
                                                 activity_threshold=activity_threshold)
                    else:
                        # Demo mode - generate random predictions
                        ic50 = np.random.lognormal(mean=np.log(500), sigma=1.0, size=1)[0]
                        result = pd.DataFrame({
                            'Predicted_IC50_nM': [ic50],
                            'Activity': ['Active' if ic50 <= activity_threshold else 'Inactive']
                        })
                    
                    result['SMILES'] = smiles
                    all_results.append(result)
                
                if all_results:
                    # Combine results
                    final_results = pd.concat(all_results, ignore_index=True)
                    
                    # Display molecule images
                    if all_molecules:
                        st.subheader("Input Molecules")
                        img = Draw.MolsToGridImage(all_molecules, molsPerRow=3, subImgSize=(200, 200))  
                        st.image(img, use_column_width=True)
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    # Format the results table
                    display_df = final_results.copy()
                    display_df['Predicted_IC50_nM'] = display_df['Predicted_IC50_nM'].round(2)
                    st.dataframe(display_df)
                    
                    # Plot IC50 values
                    if len(final_results) > 1:
                        st.subheader("IC50 Comparison")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Create plot
                        sns.barplot(x=final_results.index, y='Predicted_IC50_nM', data=final_results, 
                                    hue='Activity', palette={'Active': 'green', 'Inactive': 'red'}, ax=ax)
                        
                        # Add threshold line
                        ax.axhline(y=activity_threshold, color='black', linestyle='--', 
                                   label=f'Threshold ({activity_threshold} nM)')
                        
                        # Format plot
                        ax.set_yscale('log')
                        ax.set_xlabel('Compound')
                        ax.set_ylabel('Predicted IC50 (nM)')
                        ax.set_title('Predicted IC50 Values')
                        ax.legend()
                        
                        st.pyplot(fig)
            
            else:
                st.error("Please enter a valid SMILES string")
    
    else:  # SDF File input
        uploaded_file = st.file_uploader("Upload SDF file", type=['sdf'])
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with open('temp.sdf', 'wb') as f:
                f.write(uploaded_file.getbuffer())
                
            # Process SDF file
            descriptors_df, error = process_sdf('temp.sdf')
            
            if error:
                st.error(error)
            else:
                # Make predictions
                if model is not None:
                    results = predict_activity(model, descriptors_df, scaler, required_features, 
                                             activity_threshold=activity_threshold)
                else:
                    # Demo mode - generate random predictions
                    ic50_values = np.random.lognormal(mean=np.log(500), sigma=1.0, size=len(descriptors_df))
                    results = pd.DataFrame({
                        'Predicted_IC50_nM': ic50_values,
                        'Activity': ['Active' if ic50 <= activity_threshold else 'Inactive' for ic50 in ic50_values]
                    })
                
                # Add SMILES
                if 'smiles' in descriptors_df.columns:
                    results['SMILES'] = descriptors_df['smiles'].values
                
                # Display results
                st.subheader("Prediction Results")
                
                # Format the results table
                display_df = results.copy()
                display_df['Predicted_IC50_nM'] = display_df['Predicted_IC50_nM'].round(2)
                st.dataframe(display_df)
                
                # Plot IC50 values
                if len(results) > 1:
                    st.subheader("IC50 Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create histogram
                    sns.histplot(data=results, x='Predicted_IC50_nM', hue='Activity', 
                                 palette={'Active': 'green', 'Inactive': 'red'}, 
                                 bins=20, log_scale=True, ax=ax)
                    
                    # Add threshold line
                    ax.axvline(x=activity_threshold, color='black', linestyle='--', 
                               label=f'Threshold ({activity_threshold} nM)')
                    
                    # Format plot
                    ax.set_xlabel('Predicted IC50 (nM)')
                    ax.set_ylabel('Count')
                    ax.set_title('Distribution of Predicted IC50 Values')
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Plot structure-activity relationship
                    if 'RDKit_MolWt' in descriptors_df.columns:
                        st.subheader("Basic Structure-Activity Relationship")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Create scatter plot
                        scatter = ax.scatter(descriptors_df['RDKit_MolWt'], results['Predicted_IC50_nM'], 
                                             c=['green' if a == 'Active' else 'red' for a in results['Activity']], 
                                             alpha=0.7)
                        
                        # Add threshold line
                        ax.axhline(y=activity_threshold, color='black', linestyle='--', 
                                   label=f'Threshold ({activity_threshold} nM)')
                        
                        # Format plot
                        ax.set_yscale('log')
                        ax.set_xlabel('Molecular Weight')
                        ax.set_ylabel('Predicted IC50 (nM)')
                        ax.set_title('IC50 vs. Molecular Weight')
                        ax.legend(['Active', 'Inactive', f'Threshold ({activity_threshold} nM)'])
                        
                        st.pyplot(fig)
            
            # Clean up
            if os.path.exists('temp.sdf'):
                os.remove('temp.sdf')
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About**")
    st.sidebar.markdown("""
    This simulation tool is part of the AI-BRD4-DrugDiscovery framework,
    designed to facilitate the development of novel BRD4 inhibitors using AI/ML approaches.
    
    **Patent Pending**
    """)


if __name__ == "__main__":
    main()
