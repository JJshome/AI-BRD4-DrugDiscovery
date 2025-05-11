#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for visualizing prediction results and model performance

This module provides functions to create various plots for visualizing the
predictions of BRD4 inhibitor models and their performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


def plot_prediction_vs_actual(predictions, actual_values=None, log_scale=False, 
                             figsize=(10, 8), title=None, save_path=None):
    """
    Plot predicted IC50 values against actual values
    
    Parameters
    ----------
    predictions : array-like
        Predicted IC50 values
    actual_values : array-like or None
        Actual IC50 values (None for prediction-only plots)
    log_scale : bool
        Whether to use log scale for the axes (default: False)
    figsize : tuple
        Figure size in inches (default: (10, 8))
    title : str or None
        Plot title (default: None)
    save_path : str or None
        Path to save the plot (default: None)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if actual_values is not None:
        # Create scatter plot
        ax.scatter(actual_values, predictions, alpha=0.6, edgecolors='none')
        
        # Add identity line
        lims = [
            np.min([ax.get_xlim()[0], ax.get_ylim()[0]]),
            np.max([ax.get_xlim()[1], ax.get_ylim()[1]])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.7, label='Identity Line')
        
        # Add regression line
        m, b = np.polyfit(actual_values, predictions, 1)
        ax.plot(np.array(lims), m * np.array(lims) + b, 'r-', 
                label=f'Fit Line (slope={m:.2f})')
        
        # Calculate metrics
        r2 = r2_score(actual_values, predictions)
        rmse = np.sqrt(mean_squared_error(actual_values, predictions))
        
        # Add metrics as text
        metrics_text = f"R² = {r2:.4f}\nRMSE = {rmse:.2f} nM"
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        # Set labels
        ax.set_xlabel('Actual IC50 (nM)')
        ax.set_ylabel('Predicted IC50 (nM)')
        
    else:
        # For prediction-only plots (without actual values)
        # Create histogram
        ax.hist(predictions, bins=30, alpha=0.7, color='steelblue')
        
        # Add vertical line for activity threshold
        threshold = 1000  # Default threshold (can be parameterized)
        ax.axvline(x=threshold, color='r', linestyle='--', 
                  label=f'Activity Threshold ({threshold} nM)')
        
        # Set labels
        ax.set_xlabel('Predicted IC50 (nM)')
        ax.set_ylabel('Count')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        if actual_values is not None:
            ax.set_title('Predicted vs Actual IC50 Values')
        else:
            ax.set_title('Distribution of Predicted IC50 Values')
    
    # Set log scale if requested
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_prediction_heatmap(predictions, molecule_ids=None, sort_by='IC50', 
                          figsize=(12, 10), title=None, save_path=None):
    """
    Create a heatmap of predicted IC50 values
    
    Parameters
    ----------
    predictions : pandas.DataFrame
        DataFrame containing predictions
    molecule_ids : list or None
        List of molecule IDs (default: None)
    sort_by : str
        Column to sort by (default: 'IC50')
    figsize : tuple
        Figure size in inches (default: (12, 10))
    title : str or None
        Plot title (default: None)
    save_path : str or None
        Path to save the plot (default: None)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Prepare data
    if isinstance(predictions, pd.DataFrame):
        df = predictions.copy()
    else:
        df = pd.DataFrame(predictions)
        if molecule_ids is not None:
            df['Molecule_ID'] = molecule_ids
        
    # Ensure required columns exist
    if 'Predicted_IC50_nM' not in df.columns and 'IC50_nM' not in df.columns:
        raise ValueError("DataFrame must contain 'Predicted_IC50_nM' or 'IC50_nM' column")
    
    # Standardize column names
    if 'Predicted_IC50_nM' in df.columns:
        ic50_col = 'Predicted_IC50_nM'
    else:
        ic50_col = 'IC50_nM'
    
    # Sort by IC50 if requested
    if sort_by == 'IC50':
        df = df.sort_values(by=ic50_col)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define color map (green for active, red for inactive)
    cmap = LinearSegmentedColormap.from_list('activity', 
                                           ['forestgreen', 'khaki', 'tomato'])
    
    # Create heatmap
    if len(df) <= 50:  # For smaller datasets, show all labels
        sns.heatmap(
            df[[ic50_col]].T, 
            cmap=cmap, 
            annot=True, 
            fmt='.1f',
            linewidths=0.5,
            ax=ax
        )
    else:  # For larger datasets, skip labels to avoid overcrowding
        sns.heatmap(
            df[[ic50_col]].T, 
            cmap=cmap, 
            annot=False,
            linewidths=0.5,
            ax=ax
        )
    
    # Set labels and title
    ax.set_xlabel('Compound Index')
    ax.set_ylabel('')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('IC50 Values Heatmap')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_molecule_grid(smiles_list, values=None, labels=None, mols_per_row=4, 
                     subimg_size=(200, 200), title=None, save_path=None):
    """
    Create a grid of molecule images with optional values/labels
    
    Parameters
    ----------
    smiles_list : list
        List of SMILES strings
    values : list or None
        List of values to display (e.g., IC50 values)
    labels : list or None
        List of labels for each molecule
    mols_per_row : int
        Number of molecules per row (default: 4)
    subimg_size : tuple
        Size of each molecule image in pixels (default: (200, 200))
    title : str or None
        Plot title (default: None)
    save_path : str or None
        Path to save the plot (default: None)
        
    Returns
    -------
    PIL.Image
        Grid image of molecules
    """
    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    # Generate 2D coordinates for better visualization
    for mol in mols:
        if mol is not None:
            AllChem.Compute2DCoords(mol)
    
    # Filter out None molecules
    valid_mols = [m for m in mols if m is not None]
    
    # Prepare labels
    legends = []
    
    if values is not None and labels is None:
        # Use only values
        legends = [f"IC50: {v:.1f} nM" for i, v in enumerate(values) if mols[i] is not None]
    elif labels is not None and values is None:
        # Use only labels
        legends = [l for i, l in enumerate(labels) if mols[i] is not None]
    elif values is not None and labels is not None:
        # Use both values and labels
        legends = [f"{l}\nIC50: {v:.1f} nM" 
                 for i, (l, v) in enumerate(zip(labels, values)) if mols[i] is not None]
    
    # Create grid image
    img = Draw.MolsToGridImage(
        valid_mols, 
        molsPerRow=mols_per_row, 
        subImgSize=subimg_size,
        legends=legends if legends else None,
        useSVG=False
    )
    
    # Save image if path is provided
    if save_path:
        img.save(save_path)
    
    return img


def plot_property_activity_relationship(descriptors_df, ic50_values, property_name, 
                                      log_scale=True, activity_threshold=1000, 
                                      figsize=(10, 6), title=None, save_path=None):
    """
    Plot relationship between a molecular property and activity
    
    Parameters
    ----------
    descriptors_df : pandas.DataFrame
        DataFrame containing molecular descriptors
    ic50_values : array-like
        IC50 values
    property_name : str
        Name of the property to plot
    log_scale : bool
        Whether to use log scale for IC50 values (default: True)
    activity_threshold : float
        Threshold for classifying compounds as active/inactive (default: 1000 nM)
    figsize : tuple
        Figure size in inches (default: (10, 6))
    title : str or None
        Plot title (default: None)
    save_path : str or None
        Path to save the plot (default: None)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Check if property exists
    if property_name not in descriptors_df.columns:
        raise ValueError(f"Property '{property_name}' not found in descriptors_df")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create activity classification
    activity = ['Active' if ic50 <= activity_threshold else 'Inactive' for ic50 in ic50_values]
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Property': descriptors_df[property_name],
        'IC50': ic50_values,
        'Activity': activity
    })
    
    # Create scatter plot
    sns.scatterplot(
        data=plot_df, 
        x='Property', 
        y='IC50', 
        hue='Activity',
        palette={'Active': 'green', 'Inactive': 'red'},
        alpha=0.7,
        edgecolor='none',
        ax=ax
    )
    
    # Add horizontal line for activity threshold
    ax.axhline(y=activity_threshold, color='gray', linestyle='--', 
              label=f'Activity Threshold ({activity_threshold} nM)')
    
    # Set log scale for IC50 if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Set labels and title
    ax.set_xlabel(property_name)
    ax.set_ylabel('IC50 (nM)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Relationship between {property_name} and Activity')
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_multi_property_comparison(descriptors_df, ic50_values, property_names, 
                                 activity_threshold=1000, figsize=(15, 10), 
                                 title=None, save_path=None):
    """
    Plot multiple properties against IC50 values
    
    Parameters
    ----------
    descriptors_df : pandas.DataFrame
        DataFrame containing molecular descriptors
    ic50_values : array-like
        IC50 values
    property_names : list
        List of property names to plot
    activity_threshold : float
        Threshold for classifying compounds as active/inactive (default: 1000 nM)
    figsize : tuple
        Figure size in inches (default: (15, 10))
    title : str or None
        Plot title (default: None)
    save_path : str or None
        Path to save the plot (default: None)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Check if properties exist
    for prop in property_names:
        if prop not in descriptors_df.columns:
            raise ValueError(f"Property '{prop}' not found in descriptors_df")
    
    # Create activity classification
    activity = ['Active' if ic50 <= activity_threshold else 'Inactive' for ic50 in ic50_values]
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'IC50': ic50_values,
        'Activity': activity
    })
    
    # Add properties to DataFrame
    for prop in property_names:
        plot_df[prop] = descriptors_df[prop].values
    
    # Create figure with subplots
    n_props = len(property_names)
    n_cols = min(3, n_props)
    n_rows = (n_props + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    
    # Create scatter plots
    for i, prop in enumerate(property_names):
        row, col = i // n_cols, i % n_cols
        
        if n_rows == 1 and n_cols == 1:
            ax = axes[0]
        elif n_rows == 1:
            ax = axes[col]
        elif n_cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        
        sns.scatterplot(
            data=plot_df, 
            x=prop, 
            y='IC50', 
            hue='Activity',
            palette={'Active': 'green', 'Inactive': 'red'},
            alpha=0.7,
            edgecolor='none',
            ax=ax
        )
        
        # Add horizontal line for activity threshold
        ax.axhline(y=activity_threshold, color='gray', linestyle='--')
        
        # Set log scale for IC50
        ax.set_yscale('log')
        
        # Set labels
        ax.set_xlabel(prop)
        ax.set_ylabel('IC50 (nM)')
        ax.set_title(prop)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Remove legend (will add a single legend for the figure)
        ax.get_legend().remove()
    
    # Hide empty subplots
    for i in range(n_props, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        if n_rows == 1:
            axes[col].axis('off')
        elif n_cols == 1:
            axes[row].axis('off')
        else:
            axes[row, col].axis('off')
    
    # Add a single legend for the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
              ncol=2, frameon=True)
    
    # Set figure title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('Relationship between Molecular Properties and Activity', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_activity_distribution(ic50_values, activity_threshold=1000, 
                             figsize=(10, 6), title=None, save_path=None):
    """
    Plot distribution of IC50 values with activity classification
    
    Parameters
    ----------
    ic50_values : array-like
        IC50 values
    activity_threshold : float
        Threshold for classifying compounds as active/inactive (default: 1000 nM)
    figsize : tuple
        Figure size in inches (default: (10, 6))
    title : str or None
        Plot title (default: None)
    save_path : str or None
        Path to save the plot (default: None)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create activity classification
    activity = ['Active' if ic50 <= activity_threshold else 'Inactive' for ic50 in ic50_values]
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'IC50': ic50_values,
        'Activity': activity
    })
    
    # Create histogram with different colors for active/inactive
    sns.histplot(
        data=plot_df, 
        x='IC50', 
        hue='Activity',
        palette={'Active': 'green', 'Inactive': 'red'},
        bins=30,
        log_scale=True,
        ax=ax
    )
    
    # Add vertical line for activity threshold
    ax.axvline(x=activity_threshold, color='black', linestyle='--', 
              label=f'Activity Threshold ({activity_threshold} nM)')
    
    # Set labels and title
    ax.set_xlabel('IC50 (nM)')
    ax.set_ylabel('Count')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Distribution of IC50 Values')
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Example usage
    import numpy as np
    from rdkit import Chem
    
    # Generate random data
    np.random.seed(42)
    n_samples = 100
    
    # Random IC50 values with log-normal distribution
    actual_ic50 = np.random.lognormal(mean=np.log(500), sigma=1.0, size=n_samples)
    predicted_ic50 = actual_ic50 * (1 + np.random.normal(0, 0.3, size=n_samples))
    
    # Random molecular properties
    mol_weight = np.random.normal(350, 50, size=n_samples)
    logp = np.random.normal(3.5, 1.0, size=n_samples)
    h_donors = np.random.randint(0, 6, size=n_samples)
    h_acceptors = np.random.randint(2, 10, size=n_samples)
    
    # Create sample descriptors DataFrame
    descriptors = pd.DataFrame({
        'MW': mol_weight,
        'LogP': logp,
        'HBD': h_donors,
        'HBA': h_acceptors
    })
    
    # Sample SMILES strings
    smiles = [
        'CCO',  # Ethanol
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
        'CC(=O)NC1=CC=C(C=C1)O',  # Acetaminophen
        'C1=CC=C2C(=C1)C(=NC2)CN3CCN(CC3)C4=CC=C(C=C4)F'  # Risperidone
    ]
    
    # Plot prediction vs actual
    fig1 = plot_prediction_vs_actual(
        predicted_ic50[:20], 
        actual_ic50[:20], 
        log_scale=True,
        title='Predicted vs Actual IC50 Values (Sample)'
    )
    
    # Plot prediction heatmap
    df = pd.DataFrame({
        'Predicted_IC50_nM': predicted_ic50[:15],
        'Molecule_ID': [f'Compound_{i}' for i in range(15)]
    })
    fig2 = plot_prediction_heatmap(
        df,
        title='IC50 Values Heatmap (Sample)'
    )
    
    # Plot molecule grid
    img = plot_molecule_grid(
        smiles,
        values=predicted_ic50[:5],
        title='Sample Molecules with Predicted IC50'
    )
    
    # Plot property-activity relationship
    fig3 = plot_property_activity_relationship(
        descriptors.iloc[:30],
        predicted_ic50[:30],
        'MW',
        title='Relationship between Molecular Weight and Activity (Sample)'
    )
    
    # Plot multi-property comparison
    fig4 = plot_multi_property_comparison(
        descriptors.iloc[:40],
        predicted_ic50[:40],
        ['MW', 'LogP', 'HBD', 'HBA'],
        title='Multi-Property Comparison (Sample)'
    )
    
    # Plot activity distribution
    fig5 = plot_activity_distribution(
        predicted_ic50,
        title='Distribution of Predicted IC50 Values (Sample)'
    )
    
    plt.show()
