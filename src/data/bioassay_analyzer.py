#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for analyzing bioassay data and descriptor distributions

This module provides functions to analyze the distribution of molecular descriptors
across bioassay data, helping to understand the relationship between descriptors
and biological activity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict


def analyze_descriptor_distributions(descriptors_df, bioassay_data, target_column='IC50_nM', bins=30):
    """
    Analyze the distribution of molecular descriptors with respect to bioassay data
    
    Parameters
    ----------
    descriptors_df : pandas.DataFrame
        DataFrame containing molecular descriptors
    bioassay_data : pandas.DataFrame
        DataFrame containing bioassay data (IC50 values, etc.)
    target_column : str
        Name of the column containing the target values (default: 'IC50_nM')
    bins : int
        Number of bins for histograms (default: 30)
        
    Returns
    -------
    dict
        Dictionary containing distribution analysis results
    """
    if target_column not in bioassay_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in bioassay_data")
    
    # Ensure descriptors_df and bioassay_data have compatible indices
    if 'molecule_chembl_id' in descriptors_df.columns and 'molecule_chembl_id' in bioassay_data.columns:
        # Merge on molecule_chembl_id
        merged_df = pd.merge(descriptors_df, bioassay_data[[target_column, 'molecule_chembl_id']], 
                             on='molecule_chembl_id')
    elif set(descriptors_df.index) == set(bioassay_data.index):
        # Use index to align data
        merged_df = pd.concat([descriptors_df, bioassay_data[target_column]], axis=1)
    else:
        raise ValueError("Cannot align descriptors_df and bioassay_data. Ensure they have compatible indices or molecule_chembl_id columns.")
    
    # Select only numeric descriptor columns
    descriptor_columns = descriptors_df.select_dtypes(include=[np.number]).columns
    
    # Initialize results dictionary
    results = {
        'correlation': {},
        'mutual_information': {},
        'distribution_stats': {}
    }
    
    # Calculate correlation with target
    correlations = []
    for col in descriptor_columns:
        if col in merged_df.columns:
            corr = stats.spearmanr(merged_df[col], merged_df[target_column], nan_policy='omit')
            correlations.append((col, corr.correlation, corr.pvalue))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Store top correlated descriptors
    results['correlation'] = {col: {'correlation': corr, 'pvalue': pval} 
                             for col, corr, pval in correlations}
    
    # Calculate distribution statistics for each descriptor
    for col in descriptor_columns:
        if col in merged_df.columns:
            values = merged_df[col].dropna().values
            if len(values) > 0:
                # Basic statistics
                stats_dict = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.max(values) - np.min(values),
                    'skewness': stats.skew(values),
                    'kurtosis': stats.kurtosis(values)
                }
                
                # Store result
                results['distribution_stats'][col] = stats_dict
    
    return results


def analyze_ligand_distributions(ligands_df, descriptors_df, target_column='IC50_nM', n_bins=10):
    """
    Analyze how ligands are distributed across the descriptor space
    
    Parameters
    ----------
    ligands_df : pandas.DataFrame
        DataFrame containing ligand data (SMILES, IC50 values, etc.)
    descriptors_df : pandas.DataFrame
        DataFrame containing molecular descriptors
    target_column : str
        Name of the column containing the target values (default: 'IC50_nM')
    n_bins : int
        Number of bins for discretizing descriptor values (default: 10)
        
    Returns
    -------
    dict
        Dictionary containing ligand distribution analysis results
    """
    if target_column not in ligands_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in ligands_df")
    
    # Ensure ligands_df and descriptors_df have compatible indices
    if 'molecule_chembl_id' in descriptors_df.columns and 'molecule_chembl_id' in ligands_df.columns:
        # Merge on molecule_chembl_id
        merged_df = pd.merge(descriptors_df, ligands_df[[target_column, 'molecule_chembl_id']], 
                             on='molecule_chembl_id')
    elif set(descriptors_df.index) == set(ligands_df.index):
        # Use index to align data
        merged_df = pd.concat([descriptors_df, ligands_df[target_column]], axis=1)
    else:
        raise ValueError("Cannot align descriptors_df and ligands_df. Ensure they have compatible indices or molecule_chembl_id columns.")
    
    # Select only numeric descriptor columns
    descriptor_columns = descriptors_df.select_dtypes(include=[np.number]).columns
    
    # Initialize results
    results = {
        'bin_distributions': {},
        'activity_profiles': {}
    }
    
    # Analyze distribution of ligands across descriptor bins
    for col in descriptor_columns:
        if col in merged_df.columns:
            # Discretize descriptor values into bins
            merged_df[f'{col}_bin'] = pd.qcut(merged_df[col], q=n_bins, duplicates='drop')
            
            # Count ligands in each bin
            bin_counts = merged_df[f'{col}_bin'].value_counts().sort_index()
            
            # Get mean activity for each bin
            activity_by_bin = merged_df.groupby(f'{col}_bin')[target_column].mean().sort_index()
            
            # Store results
            results['bin_distributions'][col] = bin_counts.to_dict()
            results['activity_profiles'][col] = activity_by_bin.to_dict()
            
            # Clean up temporary column
            merged_df.drop(columns=[f'{col}_bin'], inplace=True)
    
    return results


def plot_descriptor_distributions(descriptors_df, bioassay_data, target_column='IC50_nM', 
                                 top_n=10, figsize=(15, 10)):
    """
    Plot the distribution of top molecular descriptors with respect to bioassay data
    
    Parameters
    ----------
    descriptors_df : pandas.DataFrame
        DataFrame containing molecular descriptors
    bioassay_data : pandas.DataFrame
        DataFrame containing bioassay data (IC50 values, etc.)
    target_column : str
        Name of the column containing the target values (default: 'IC50_nM')
    top_n : int
        Number of top correlated descriptors to plot (default: 10)
    figsize : tuple
        Figure size in inches (default: (15, 10))
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plots
    """
    # Analyze descriptor distributions
    analysis_results = analyze_descriptor_distributions(descriptors_df, bioassay_data, target_column)
    
    # Sort descriptors by absolute correlation
    correlations = [(col, abs(val['correlation'])) 
                    for col, val in analysis_results['correlation'].items()]
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Select top correlated descriptors
    top_descriptors = [col for col, _ in correlations[:top_n]]
    
    # Ensure descriptors_df and bioassay_data have compatible indices
    if 'molecule_chembl_id' in descriptors_df.columns and 'molecule_chembl_id' in bioassay_data.columns:
        # Merge on molecule_chembl_id
        merged_df = pd.merge(descriptors_df, bioassay_data[[target_column, 'molecule_chembl_id']], 
                             on='molecule_chembl_id')
    elif set(descriptors_df.index) == set(bioassay_data.index):
        # Use index to align data
        merged_df = pd.concat([descriptors_df, bioassay_data[target_column]], axis=1)
    else:
        raise ValueError("Cannot align descriptors_df and bioassay_data. Ensure they have compatible indices or molecule_chembl_id columns.")
    
    # Define activity threshold (for visualization)
    activity_threshold = np.median(merged_df[target_column])
    merged_df['Activity'] = merged_df[target_column].apply(
        lambda x: 'Active' if x <= activity_threshold else 'Inactive')
    
    # Create figure
    fig, axes = plt.subplots(nrows=len(top_descriptors), ncols=2, figsize=figsize)
    
    # Plot each descriptor
    for i, descriptor in enumerate(top_descriptors):
        if descriptor in merged_df.columns:
            # Histogram by activity
            sns.histplot(data=merged_df, x=descriptor, hue='Activity', ax=axes[i, 0], 
                         palette={'Active': 'green', 'Inactive': 'red'})
            axes[i, 0].set_title(f"{descriptor} Distribution")
            
            # Scatter plot against activity
            sns.scatterplot(data=merged_df, x=descriptor, y=target_column, hue='Activity', 
                            ax=axes[i, 1], palette={'Active': 'green', 'Inactive': 'red'})
            axes[i, 1].set_title(f"{descriptor} vs {target_column}")
            axes[i, 1].set_yscale('log')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def analyze_distributions(data, target_column='IC50_nM', numeric_only=True):
    """
    Analyze the distribution of all columns in a DataFrame with respect to a target column
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing features and target values
    target_column : str
        Name of the column containing the target values (default: 'IC50_nM')
    numeric_only : bool
        Whether to include only numeric columns (default: True)
        
    Returns
    -------
    dict
        Dictionary containing distribution analysis results
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Select columns for analysis
    if numeric_only:
        feature_columns = data.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in feature_columns if col != target_column]
    else:
        feature_columns = [col for col in data.columns if col != target_column]
    
    # Initialize results
    results = defaultdict(dict)
    
    # For each feature column
    for col in feature_columns:
        # Skip columns with all missing values
        if data[col].isna().all():
            continue
        
        # Analyze numeric features
        if np.issubdtype(data[col].dtype, np.number):
            # Correlation
            corr = stats.spearmanr(data[col], data[target_column], nan_policy='omit')
            results[col]['correlation'] = corr.correlation
            results[col]['correlation_pvalue'] = corr.pvalue
            
            # Basic statistics
            results[col]['mean'] = data[col].mean()
            results[col]['median'] = data[col].median()
            results[col]['std'] = data[col].std()
            results[col]['min'] = data[col].min()
            results[col]['max'] = data[col].max()
            
            # Bin-based analysis
            try:
                bins = pd.qcut(data[col], q=10, duplicates='drop')
                group_means = data.groupby(bins)[target_column].mean()
                results[col]['bin_means'] = group_means.to_dict()
                
                # Trend analysis
                bin_values = list(group_means.values)
                if len(bin_values) >= 3:  # Need at least 3 points for trend analysis
                    # Test for monotonicity (Spearman's rank correlation)
                    trend_corr = stats.spearmanr(range(len(bin_values)), bin_values).correlation
                    results[col]['trend'] = trend_corr
                    
                    # Categorize trend
                    if abs(trend_corr) < 0.3:
                        results[col]['trend_type'] = 'No trend'
                    elif trend_corr > 0:
                        results[col]['trend_type'] = 'Increasing'
                    else:  # trend_corr < 0
                        results[col]['trend_type'] = 'Decreasing'
            except:
                # Some features may not be amenable to binning
                pass
        
        # Analyze categorical features
        else:
            # Count distribution
            value_counts = data[col].value_counts()
            results[col]['value_counts'] = value_counts.to_dict()
            
            # Group-based analysis
            group_means = data.groupby(col)[target_column].mean()
            results[col]['group_means'] = group_means.to_dict()
            
            # ANOVA or Kruskal-Wallis test
            groups = [data[data[col] == val][target_column].dropna().values 
                     for val in data[col].unique()]
            groups = [g for g in groups if len(g) > 0]  # Remove empty groups
            
            if len(groups) >= 2:  # Need at least 2 groups for comparison
                try:
                    # Try non-parametric test first (more robust)
                    _, pval = stats.kruskal(*groups)
                    results[col]['group_diff_pvalue'] = pval
                    results[col]['group_diff_test'] = 'Kruskal-Wallis'
                except:
                    try:
                        # Fall back to ANOVA if Kruskal-Wallis fails
                        _, pval = stats.f_oneway(*groups)
                        results[col]['group_diff_pvalue'] = pval
                        results[col]['group_diff_test'] = 'ANOVA'
                    except:
                        # Some features may not be amenable to statistical testing
                        pass
    
    return dict(results)  # Convert defaultdict to regular dict


if __name__ == "__main__":
    # Example usage
    from ligand_selection import select_brd4_ligands
    from descriptor_extraction import extract_descriptors
    
    print("Selecting BRD4 ligands from ChEMBL...")
    ligands = select_brd4_ligands(target="BRD4", ic50_max=1000)
    print(f"Found {len(ligands)} BRD4 ligands")
    
    # Extract a small subset for testing
    test_ligands = ligands.head(100)
    
    print("Extracting molecular descriptors...")
    descriptors_df = extract_descriptors(
        test_ligands, 
        descriptor_types=['constitutional', 'topological']
    )
    
    print("Analyzing descriptor distributions...")
    analysis = analyze_descriptor_distributions(
        descriptors_df, 
        test_ligands,
        target_column='IC50_nM'
    )
    
    print("\nTop 5 correlated descriptors:")
    corr_items = sorted(analysis['correlation'].items(), 
                        key=lambda x: abs(x[1]['correlation']), reverse=True)
    for descriptor, stats in corr_items[:5]:
        print(f"{descriptor}: correlation={stats['correlation']:.4f}, p-value={stats['pvalue']:.4e}")
    
    print("\nAnalyzing ligand distributions...")
    ligand_dist = analyze_ligand_distributions(
        test_ligands,
        descriptors_df,
        target_column='IC50_nM'
    )
    
    print("\nPlotting descriptor distributions...")
    fig = plot_descriptor_distributions(
        descriptors_df,
        test_ligands,
        target_column='IC50_nM',
        top_n=3
    )
    plt.show()
