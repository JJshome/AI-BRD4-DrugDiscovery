#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for feature selection techniques applied to molecular descriptor data

This module provides functions to select the most informative molecular descriptors
for predicting IC50 values, using various feature selection methods.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings


def preprocess_for_feature_selection(X, y, scale=True, handle_na='median'):
    """
    Preprocess data for feature selection
    
    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame containing features (molecular descriptors)
    y : pandas.Series or numpy.ndarray
        Target values (IC50)
    scale : bool
        Whether to standardize features (default: True)
    handle_na : str
        Method to handle missing values ('median', 'mean', 'drop')
        
    Returns
    -------
    numpy.ndarray
        Preprocessed features
    numpy.ndarray
        Preprocessed target values
    list
        Feature names
    """
    # Get feature names
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Convert to numpy arrays
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    y_array = y.values if isinstance(y, pd.Series) else y
    
    # Handle missing values
    if handle_na == 'median':
        if isinstance(X, pd.DataFrame):
            X_filled = X.fillna(X.median())
            X_array = X_filled.values
        else:
            # For numpy arrays, compute median column-wise
            median_values = np.nanmedian(X_array, axis=0)
            mask = np.isnan(X_array)
            X_array = np.copy(X_array)  # Create a copy to avoid modifying the original
            for i in range(X_array.shape[1]):
                X_array[mask[:, i], i] = median_values[i]
    
    elif handle_na == 'mean':
        if isinstance(X, pd.DataFrame):
            X_filled = X.fillna(X.mean())
            X_array = X_filled.values
        else:
            # For numpy arrays, compute mean column-wise
            mean_values = np.nanmean(X_array, axis=0)
            mask = np.isnan(X_array)
            X_array = np.copy(X_array)
            for i in range(X_array.shape[1]):
                X_array[mask[:, i], i] = mean_values[i]
    
    elif handle_na == 'drop':
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            # Get rows with no NaN values
            mask = ~X.isna().any(axis=1)
            X_array = X[mask].values
            y_array = y[mask].values
        else:
            # For numpy arrays
            mask = ~np.isnan(X_array).any(axis=1)
            X_array = X_array[mask]
            y_array = y_array[mask]
    
    elif handle_na is not None:
        raise ValueError(f"Unknown NaN handling method: {handle_na}")
    
    # Scale features
    if scale:
        scaler = StandardScaler()
        X_array = scaler.fit_transform(X_array)
    
    return X_array, y_array, feature_names


def select_features_correlation(X, y, n_features=20, method='pearson', abs_correlation=True):
    """
    Select features based on correlation with target
    
    Parameters
    ----------
    X : array-like
        Features (molecular descriptors)
    y : array-like
        Target values (IC50)
    n_features : int
        Number of features to select (default: 20)
    method : str
        Correlation method ('pearson', 'spearman', 'kendall')
    abs_correlation : bool
        Whether to use absolute correlation values (default: True)
        
    Returns
    -------
    list
        Indices of selected features
    list
        Correlation scores for all features
    """
    X_array, y_array, feature_names = preprocess_for_feature_selection(X, y)
    
    # Calculate correlation for each feature
    correlations = []
    for i in range(X_array.shape[1]):
        if method == 'pearson':
            corr, _ = stats.pearsonr(X_array[:, i], y_array)
        elif method == 'spearman':
            corr, _ = stats.spearmanr(X_array[:, i], y_array)
        elif method == 'kendall':
            corr, _ = stats.kendalltau(X_array[:, i], y_array)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        correlations.append(corr)
    
    # Convert to numpy array
    correlations = np.array(correlations)
    
    # Sort by absolute correlation if requested
    if abs_correlation:
        feature_order = np.argsort(np.abs(correlations))[::-1]
    else:
        feature_order = np.argsort(correlations)[::-1]
    
    # Select top features
    selected_indices = feature_order[:n_features]
    
    return selected_indices, correlations


def select_features_mutual_info(X, y, n_features=20):
    """
    Select features based on mutual information with target
    
    Parameters
    ----------
    X : array-like
        Features (molecular descriptors)
    y : array-like
        Target values (IC50)
    n_features : int
        Number of features to select (default: 20)
        
    Returns
    -------
    list
        Indices of selected features
    list
        Mutual information scores for all features
    """
    X_array, y_array, feature_names = preprocess_for_feature_selection(X, y)
    
    # Use SelectKBest with mutual information
    selector = SelectKBest(mutual_info_regression, k=n_features)
    selector.fit(X_array, y_array)
    
    # Get selected features and scores
    selected_indices = np.argsort(selector.scores_)[::-1][:n_features]
    
    return selected_indices, selector.scores_


def select_features_rfe(X, y, n_features=20, estimator='rf', step=0.1, cv=None):
    """
    Select features using Recursive Feature Elimination (RFE)
    
    Parameters
    ----------
    X : array-like
        Features (molecular descriptors)
    y : array-like
        Target values (IC50)
    n_features : int
        Number of features to select (default: 20)
    estimator : str or object
        Estimator to use ('rf', 'gbm', 'lasso', 'elasticnet' or a scikit-learn estimator)
    step : float or int
        Step size for RFE (default: 0.1, meaning 10% of features are removed at each step)
    cv : int or None
        Number of cross-validation folds (None for no cross-validation)
        
    Returns
    -------
    list
        Indices of selected features
    object
        Trained RFE or RFECV object
    """
    X_array, y_array, feature_names = preprocess_for_feature_selection(X, y)
    
    # Create estimator
    if estimator == 'rf':
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    elif estimator == 'gbm':
        estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif estimator == 'lasso':
        estimator = Lasso(alpha=0.1, random_state=42)
    elif estimator == 'elasticnet':
        estimator = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    elif not hasattr(estimator, 'fit'):
        raise ValueError(f"Unknown estimator: {estimator}")
    
    # Create RFE object
    if cv is not None:
        selector = RFECV(estimator, step=step, cv=cv, scoring='neg_mean_squared_error')
    else:
        selector = RFE(estimator, n_features_to_select=n_features, step=step)
    
    # Fit RFE
    selector.fit(X_array, y_array)
    
    # Get selected features
    selected_indices = np.where(selector.support_)[0]
    
    return selected_indices, selector


def select_features_model_based(X, y, n_features=20, estimator='rf', threshold=None):
    """
    Select features using model-based feature importance
    
    Parameters
    ----------
    X : array-like
        Features (molecular descriptors)
    y : array-like
        Target values (IC50)
    n_features : int
        Number of features to select (default: 20)
    estimator : str or object
        Estimator to use ('rf', 'gbm', 'lasso', 'elasticnet' or a scikit-learn estimator)
    threshold : float or None
        Threshold for feature selection (None to select n_features)
        
    Returns
    -------
    list
        Indices of selected features
    object
        Trained SelectFromModel object
    """
    X_array, y_array, feature_names = preprocess_for_feature_selection(X, y)
    
    # Create estimator
    if estimator == 'rf':
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    elif estimator == 'gbm':
        estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif estimator == 'lasso':
        estimator = Lasso(alpha=0.1, random_state=42)
    elif estimator == 'elasticnet':
        estimator = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    elif not hasattr(estimator, 'fit'):
        raise ValueError(f"Unknown estimator: {estimator}")
    
    # Create selector
    if threshold is not None:
        selector = SelectFromModel(estimator, threshold=threshold)
    else:
        # Train model first to get feature importances
        estimator.fit(X_array, y_array)
        
        # Get feature importances
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            importances = np.abs(estimator.coef_)
        else:
            raise ValueError("Estimator does not have feature_importances_ or coef_ attribute")
        
        # Calculate threshold to select n_features
        sorted_importances = np.sort(importances)[::-1]
        if len(sorted_importances) > n_features:
            threshold = sorted_importances[n_features - 1]
        else:
            threshold = sorted_importances[-1]
        
        selector = SelectFromModel(estimator, threshold=threshold)
    
    # Fit selector
    selector.fit(X_array, y_array)
    
    # Get selected features
    selected_indices = np.where(selector.get_support())[0]
    
    return selected_indices, selector


def select_features(X, y, method='correlation', n_features=20, **kwargs):
    """
    Select features using various methods
    
    Parameters
    ----------
    X : array-like
        Features (molecular descriptors)
    y : array-like
        Target values (IC50)
    method : str
        Feature selection method ('correlation', 'mutual_info', 'rfe', 'model_based')
    n_features : int
        Number of features to select (default: 20)
    **kwargs : dict
        Additional parameters for the feature selection method
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        Selected features
    list
        Names of selected features
    """
    # Convert input to DataFrame/Series if needed
    if not isinstance(X, pd.DataFrame):
        # Convert to DataFrame with default column names
        X = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
    
    if not isinstance(y, pd.Series) and not isinstance(y, np.ndarray):
        y = np.array(y)
    
    # Select features using the specified method
    if method == 'correlation':
        correlation_method = kwargs.get('correlation_method', 'pearson')
        abs_correlation = kwargs.get('abs_correlation', True)
        indices, scores = select_features_correlation(
            X, y, n_features, correlation_method, abs_correlation)
    
    elif method == 'mutual_info':
        indices, scores = select_features_mutual_info(X, y, n_features)
    
    elif method == 'rfe':
        estimator = kwargs.get('estimator', 'rf')
        step = kwargs.get('step', 0.1)
        cv = kwargs.get('cv', None)
        indices, _ = select_features_rfe(X, y, n_features, estimator, step, cv)
    
    elif method == 'model_based':
        estimator = kwargs.get('estimator', 'rf')
        threshold = kwargs.get('threshold', None)
        indices, _ = select_features_model_based(X, y, n_features, estimator, threshold)
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    # Get feature names
    feature_names = X.columns[indices].tolist()
    
    # Return selected features as DataFrame or numpy array
    if isinstance(X, pd.DataFrame):
        selected_features = X.iloc[:, indices]
    else:
        selected_features = X[:, indices]
    
    return selected_features, feature_names


def evaluate_feature_selection(X, y, methods=['correlation', 'mutual_info', 'rfe', 'model_based'], 
                              n_features_list=[10, 20, 50, 100, 200], cv=5):
    """
    Evaluate different feature selection methods and number of features
    
    Parameters
    ----------
    X : array-like
        Features (molecular descriptors)
    y : array-like
        Target values (IC50)
    methods : list
        List of feature selection methods to evaluate
    n_features_list : list
        List of numbers of features to evaluate
    cv : int
        Number of cross-validation folds
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing evaluation results
    """
    # Initialize results
    results = []
    
    # Preprocess data
    X_array, y_array, feature_names = preprocess_for_feature_selection(X, y)
    
    # Create base models for evaluation
    models = {
        'rf': RandomForestRegressor(n_estimators=100, random_state=42),
        'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    }
    
    # Evaluate each method and number of features
    for method in methods:
        for n_features in n_features_list:
            # Skip if n_features > number of available features
            if n_features > X_array.shape[1]:
                continue
            
            try:
                # Select features
                selected_X, _ = select_features(X_array, y_array, method=method, n_features=n_features)
                
                # Evaluate with each model
                for model_name, model in models.items():
                    # Cross-validation
                    scores = cross_val_score(
                        model, selected_X, y_array, 
                        cv=cv, scoring='neg_mean_squared_error'
                    )
                    
                    # Calculate RMSE
                    rmse_scores = np.sqrt(-scores)
                    mean_rmse = np.mean(rmse_scores)
                    std_rmse = np.std(rmse_scores)
                    
                    # Store results
                    results.append({
                        'Method': method,
                        'n_features': n_features,
                        'Model': model_name,
                        'Mean_RMSE': mean_rmse,
                        'Std_RMSE': std_rmse
                    })
            except Exception as e:
                warnings.warn(f"Error evaluating {method} with {n_features} features: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    
    # Create a sample dataset
    X, y = make_regression(n_samples=1000, n_features=100, noise=0.1, random_state=42)
    
    # Test correlation-based feature selection
    print("Testing correlation-based feature selection...")
    corr_indices, corr_scores = select_features_correlation(X, y, n_features=10)
    print(f"Selected feature indices: {corr_indices}")
    print(f"Top 5 correlation scores: {corr_scores[corr_indices[:5]]}")
    
    # Test mutual information-based feature selection
    print("\nTesting mutual information-based feature selection...")
    mi_indices, mi_scores = select_features_mutual_info(X, y, n_features=10)
    print(f"Selected feature indices: {mi_indices}")
    print(f"Top 5 mutual information scores: {mi_scores[mi_indices[:5]]}")
    
    # Test RFE-based feature selection
    print("\nTesting RFE-based feature selection...")
    rfe_indices, rfe_selector = select_features_rfe(X, y, n_features=10)
    print(f"Selected feature indices: {rfe_indices}")
    
    # Test model-based feature selection
    print("\nTesting model-based feature selection...")
    model_indices, model_selector = select_features_model_based(X, y, n_features=10)
    print(f"Selected feature indices: {model_indices}")
    
    # Evaluate different feature selection methods
    print("\nEvaluating feature selection methods...")
    results = evaluate_feature_selection(
        X, y, 
        methods=['correlation', 'mutual_info'], 
        n_features_list=[5, 10, 20],
        cv=3
    )
    print(results)
