#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Outlier Detection Module

This module implements methods for detecting and removing outliers in molecular descriptor data.
According to the patent, this is a critical step in the AI-based drug development method
as it helps improve the quality of data for regression analysis.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import List, Dict, Optional, Union, Tuple, Set

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OutlierDetector:
    """Class for detecting and removing outliers in molecular descriptor data."""
    
    def __init__(self, method='zscore', contamination=0.05):
        """
        Initialize the OutlierDetector object.
        
        Args:
            method (str, optional): Method for outlier detection. 
                                   Options: 'zscore', 'iqr', 'isolation_forest', 'lof', 'elliptic'. 
                                   Defaults to 'zscore'.
            contamination (float, optional): Expected proportion of outliers in the dataset. 
                                           Used for isolation_forest, lof, and elliptic methods.
                                           Defaults to 0.05.
        """
        self.method = method
        self.contamination = contamination
        self.outlier_indices = None
        self.feature_outliers = None
        self.molecular_outliers = None
    
    def detect_feature_outliers(self, df, feature_cols=None, z_threshold=3.0, iqr_multiplier=1.5):
        """
        Detect outliers in each feature/descriptor column.
        
        Args:
            df (pandas.DataFrame): DataFrame containing the data.
            feature_cols (list, optional): List of feature columns to check for outliers. 
                                         If None, all numeric columns are used.
            z_threshold (float, optional): Z-score threshold for 'zscore' method. Defaults to 3.0.
            iqr_multiplier (float, optional): IQR multiplier for 'iqr' method. Defaults to 1.5.
            
        Returns:
            dict: Dictionary mapping column names to lists of outlier indices.
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            # Exclude any ID columns
            feature_cols = [col for col in feature_cols if not any(x in col.lower() for x in ['id', 'idx', 'index'])]
        
        logger.info(f"Detecting outliers in {len(feature_cols)} feature columns using {self.method} method.")
        
        outlier_indices = {}
        
        for col in feature_cols:
            # Skip columns with all NaN or constant values
            if df[col].isna().all() or df[col].nunique() <= 1:
                continue
            
            # Get non-null values for the column
            values = df[col].dropna().values
            
            if self.method == 'zscore':
                z_scores = stats.zscore(values, nan_policy='omit')
                outliers = np.abs(z_scores) > z_threshold
                outlier_idx = np.where(outliers)[0]
            
            elif self.method == 'iqr':
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - iqr_multiplier * iqr
                upper_bound = q3 + iqr_multiplier * iqr
                outliers = (values < lower_bound) | (values > upper_bound)
                outlier_idx = np.where(outliers)[0]
            
            else:
                logger.warning(f"Method {self.method} not supported for feature outlier detection. Using zscore.")
                z_scores = stats.zscore(values, nan_policy='omit')
                outliers = np.abs(z_scores) > z_threshold
                outlier_idx = np.where(outliers)[0]
            
            # Map back to original indices
            if len(outlier_idx) > 0:
                original_indices = df[~df[col].isna()].index.values[outlier_idx]
                outlier_indices[col] = original_indices.tolist()
        
        self.feature_outliers = outlier_indices
        
        # Log summary
        total_feature_outliers = set()
        for indices in outlier_indices.values():
            total_feature_outliers.update(indices)
        
        logger.info(f"Detected {len(total_feature_outliers)} unique outlier rows across all features.")
        return outlier_indices
    
    def detect_molecular_outliers(self, df, feature_cols=None, distance_metric='euclidean', random_state=42):
        """
        Detect outliers considering the entire molecular structure.
        
        Args:
            df (pandas.DataFrame): DataFrame containing the data.
            feature_cols (list, optional): List of feature columns to use. If None, all numeric columns are used.
            distance_metric (str, optional): Distance metric for LOF method. Defaults to 'euclidean'.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
            
        Returns:
            list: List of outlier indices.
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            # Exclude any ID columns
            feature_cols = [col for col in feature_cols if not any(x in col.lower() for x in ['id', 'idx', 'index'])]
        
        # Get data without NaN values
        data = df[feature_cols].fillna(df[feature_cols].mean())
        
        logger.info(f"Detecting molecular outliers using {self.method} method with {len(feature_cols)} features.")
        
        if self.method == 'isolation_forest':
            clf = IsolationForest(contamination=self.contamination, random_state=random_state)
            y_pred = clf.fit_predict(data)
            outliers = y_pred == -1
        
        elif self.method == 'lof':
            clf = LocalOutlierFactor(n_neighbors=20, contamination=self.contamination, metric=distance_metric)
            y_pred = clf.fit_predict(data)
            outliers = y_pred == -1
        
        elif self.method == 'elliptic':
            clf = EllipticEnvelope(contamination=self.contamination, random_state=random_state)
            y_pred = clf.fit_predict(data)
            outliers = y_pred == -1
        
        else:
            logger.warning(f"Method {self.method} not supported for molecular outlier detection. Using isolation_forest.")
            clf = IsolationForest(contamination=self.contamination, random_state=random_state)
            y_pred = clf.fit_predict(data)
            outliers = y_pred == -1
        
        outlier_indices = np.where(outliers)[0].tolist()
        self.molecular_outliers = outlier_indices
        
        logger.info(f"Detected {len(outlier_indices)} molecular outliers.")
        return outlier_indices
    
    def remove_outliers(self, df, feature_outliers=True, molecular_outliers=True, min_features_for_removal=1):
        """
        Remove detected outliers from the DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame containing the data.
            feature_outliers (bool, optional): Whether to remove feature outliers. Defaults to True.
            molecular_outliers (bool, optional): Whether to remove molecular outliers. Defaults to True.
            min_features_for_removal (int, optional): Minimum number of features that must flag a row as an outlier
                                                    for it to be removed. Defaults to 1.
            
        Returns:
            pandas.DataFrame: DataFrame with outliers removed.
        """
        outlier_indices = set()
        
        # Add feature outliers
        if feature_outliers and self.feature_outliers is not None:
            # Count how many features flag each row as an outlier
            outlier_counts = {}
            for indices in self.feature_outliers.values():
                for idx in indices:
                    outlier_counts[idx] = outlier_counts.get(idx, 0) + 1
            
            # Add indices that meet the minimum threshold
            outlier_indices.update({idx for idx, count in outlier_counts.items() if count >= min_features_for_removal})
        
        # Add molecular outliers
        if molecular_outliers and self.molecular_outliers is not None:
            outlier_indices.update(self.molecular_outliers)
        
        # Remove outliers
        if outlier_indices:
            clean_df = df.drop(index=list(outlier_indices))
            logger.info(f"Removed {len(outlier_indices)} outliers. Remaining rows: {len(clean_df)}")
            return clean_df
        else:
            logger.info("No outliers to remove.")
            return df
    
    def detect_and_remove_outliers(self, df, feature_cols=None, target_col=None, 
                                   feature_outliers=True, molecular_outliers=True, 
                                   min_features_for_removal=1, z_threshold=3.0, iqr_multiplier=1.5,
                                   distance_metric='euclidean', random_state=42):
        """
        Detect and remove outliers in one step.
        
        Args:
            df (pandas.DataFrame): DataFrame containing the data.
            feature_cols (list, optional): List of feature columns to use. If None, all numeric columns are used.
            target_col (str, optional): Target column name. If provided, also checks for outliers in this column.
            feature_outliers (bool, optional): Whether to detect feature outliers. Defaults to True.
            molecular_outliers (bool, optional): Whether to detect molecular outliers. Defaults to True.
            min_features_for_removal (int, optional): Minimum number of features that must flag a row as an outlier.
            z_threshold (float, optional): Z-score threshold for 'zscore' method. Defaults to 3.0.
            iqr_multiplier (float, optional): IQR multiplier for 'iqr' method. Defaults to 1.5.
            distance_metric (str, optional): Distance metric for LOF method. Defaults to 'euclidean'.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
            
        Returns:
            pandas.DataFrame: DataFrame with outliers removed.
        """
        # Identify feature columns if not provided
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            # Exclude any ID columns
            feature_cols = [col for col in feature_cols if not any(x in col.lower() for x in ['id', 'idx', 'index'])]
            
            # Remove target column from feature columns if provided
            if target_col is not None and target_col in feature_cols:
                feature_cols.remove(target_col)
        
        # Add target column for outlier detection if provided
        detection_cols = feature_cols.copy()
        if target_col is not None and target_col not in detection_cols:
            detection_cols.append(target_col)
        
        # Detect feature outliers
        if feature_outliers:
            self.detect_feature_outliers(df, detection_cols, z_threshold, iqr_multiplier)
        
        # Detect molecular outliers
        if molecular_outliers:
            self.detect_molecular_outliers(df, feature_cols, distance_metric, random_state)
        
        # Remove outliers
        return self.remove_outliers(df, feature_outliers, molecular_outliers, min_features_for_removal)
    
    def visualize_outliers(self, df, feature_cols=None, target_col=None, n_features=5, 
                          feature_outliers=True, molecular_outliers=True, save_path=None):
        """
        Visualize detected outliers.
        
        Args:
            df (pandas.DataFrame): DataFrame containing the data.
            feature_cols (list, optional): List of feature columns to visualize. If None, all numeric columns are used.
            target_col (str, optional): Target column name for color coding. Defaults to None.
            n_features (int, optional): Number of most affected features to visualize. Defaults to 5.
            feature_outliers (bool, optional): Whether to include feature outliers. Defaults to True.
            molecular_outliers (bool, optional): Whether to include molecular outliers. Defaults to True.
            save_path (str, optional): Path to save the visualization. Defaults to None.
            
        Returns:
            matplotlib.figure.Figure: Generated figure.
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            # Exclude any ID columns
            feature_cols = [col for col in feature_cols if not any(x in col.lower() for x in ['id', 'idx', 'index'])]
        
        # Collect all outlier indices
        outlier_indices = set()
        if feature_outliers and self.feature_outliers is not None:
            for indices in self.feature_outliers.values():
                outlier_indices.update(indices)
        
        if molecular_outliers and self.molecular_outliers is not None:
            outlier_indices.update(self.molecular_outliers)
        
        # If no outliers detected, return
        if not outlier_indices:
            logger.warning("No outliers to visualize.")
            return None
        
        # Select most affected features (those with most outliers)
        if feature_outliers and self.feature_outliers is not None:
            feature_count = {col: len(indices) for col, indices in self.feature_outliers.items()}
            top_features = sorted(feature_count.keys(), key=lambda x: feature_count[x], reverse=True)[:n_features]
        else:
            top_features = feature_cols[:n_features]
        
        # Create a figure
        fig, axes = plt.subplots(len(top_features), 1, figsize=(12, 4 * len(top_features)))
        
        if len(top_features) == 1:
            axes = [axes]
        
        # Create color mapper if target column is provided
        if target_col is not None and target_col in df.columns:
            color_mapper = df[target_col]
        else:
            color_mapper = None
        
        # Plot histograms/boxplots for each feature with outliers highlighted
        for i, feature in enumerate(top_features):
            ax = axes[i]
            
            # Create a mask for outliers in this feature
            outlier_mask = np.zeros(len(df), dtype=bool)
            if feature_outliers and self.feature_outliers is not None and feature in self.feature_outliers:
                for idx in self.feature_outliers[feature]:
                    if idx < len(outlier_mask):
                        outlier_mask[idx] = True
            
            # Boxplot
            sns.boxplot(x=df[feature], ax=ax)
            
            # Scatter plot of outliers
            if any(outlier_mask):
                outlier_values = df.loc[outlier_mask, feature]
                ax.scatter(outlier_values, np.zeros_like(outlier_values), color='r', s=50, alpha=0.7, label='Outliers')
            
            ax.set_title(f"{feature} Distribution with Outliers")
            ax.set_xlabel(feature)
            ax.legend()
        
        plt.tight_layout()
        
        # Save the figure if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Saved outlier visualization to {save_path}")
        
        return fig
    
    def save_outlier_report(self, df, path, feature_cols=None, include_values=True):
        """
        Save a detailed report of detected outliers.
        
        Args:
            df (pandas.DataFrame): DataFrame containing the data.
            path (str): Path to save the report.
            feature_cols (list, optional): List of feature columns. If None, all numeric columns are used.
            include_values (bool, optional): Whether to include the actual values of outliers. Defaults to True.
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            # Exclude any ID columns
            feature_cols = [col for col in feature_cols if not any(x in col.lower() for x in ['id', 'idx', 'index'])]
        
        # Collect all outlier indices
        all_outliers = set()
        feature_specific_outliers = {}
        
        if self.feature_outliers is not None:
            for feature, indices in self.feature_outliers.items():
                if indices:
                    feature_specific_outliers[feature] = indices
                    all_outliers.update(indices)
        
        if self.molecular_outliers is not None:
            all_outliers.update(self.molecular_outliers)
        
        # Generate report
        report = []
        report.append(f"Outlier Detection Report\n")
        report.append(f"Method: {self.method}")
        report.append(f"Total unique outliers: {len(all_outliers)}")
        report.append(f"Feature outliers: {len(feature_specific_outliers)}")
        report.append(f"Molecular outliers: {len(self.molecular_outliers) if self.molecular_outliers is not None else 0}")
        report.append("\n\nFeature-specific outliers:")
        
        # Report for each feature
        for feature, indices in feature_specific_outliers.items():
            report.append(f"\n{feature}: {len(indices)} outliers")
            
            if include_values and indices:
                # Get statistics for this feature
                feature_stats = df[feature].describe()
                report.append(f"Feature statistics:")
                report.append(f"  Mean: {feature_stats['mean']:.4f}")
                report.append(f"  Std: {feature_stats['std']:.4f}")
                report.append(f"  Min: {feature_stats['min']:.4f}")
                report.append(f"  25%: {feature_stats['25%']:.4f}")
                report.append(f"  50%: {feature_stats['50%']:.4f}")
                report.append(f"  75%: {feature_stats['75%']:.4f}")
                report.append(f"  Max: {feature_stats['max']:.4f}")
                
                # Get outlier values
                report.append(f"Outlier values:")
                outlier_values = df.loc[indices, feature].values
                for i, val in enumerate(outlier_values[:10]):  # Limit to first 10 for brevity
                    report.append(f"  {i+1}. {val:.4f}")
                
                if len(outlier_values) > 10:
                    report.append(f"  ... and {len(outlier_values) - 10} more")
        
        # Save the report
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Saved outlier report to {path}")

# Example usage
if __name__ == "__main__":
    # Example with sample data
    from src.data.ligand_selection import LigandSelector
    from src.data.descriptor_extraction import DescriptorExtractor
    
    # Load ligands and descriptors
    selector = LigandSelector()
    ligands = selector.load_ligands("data/brd4_ligands.csv")
    
    extractor = DescriptorExtractor()
    descriptors = extractor.load_descriptors("data/brd4_descriptors_filtered.csv")
    
    # Detect and remove outliers
    detector = OutlierDetector(method='isolation_forest', contamination=0.05)
    
    # Full process
    clean_data = detector.detect_and_remove_outliers(
        descriptors,
        target_col="standard_value",
        feature_outliers=True,
        molecular_outliers=True,
        min_features_for_removal=2
    )
    
    # Save clean data
    clean_data.to_csv("data/brd4_descriptors_clean.csv", index=False)
    
    # Visualize outliers
    detector.visualize_outliers(
        descriptors,
        target_col="standard_value",
        n_features=5,
        save_path="results/outlier_visualization.png"
    )
    
    # Save outlier report
    detector.save_outlier_report(
        descriptors,
        path="results/outlier_report.txt",
        include_values=True
    )
