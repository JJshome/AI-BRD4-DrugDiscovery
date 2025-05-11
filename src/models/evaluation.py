#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for evaluating machine learning models for predicting IC50 values

This module provides functions to evaluate and visualize the performance of
regression models trained to predict IC50 values of BRD4 inhibitors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import cross_val_predict, learning_curve


def calculate_metrics(y_true, y_pred, metrics=None):
    """
    Calculate performance metrics for regression predictions
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    metrics : list or None
        List of metrics to calculate (None for all)
        Options: 'rmse', 'mae', 'r2', 'mape', 'explained_variance'
        
    Returns
    -------
    dict
        Dictionary containing calculated metrics
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'r2', 'mape', 'explained_variance']
    
    results = {}
    
    # Root Mean Squared Error
    if 'rmse' in metrics:
        results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Mean Absolute Error
    if 'mae' in metrics:
        results['mae'] = mean_absolute_error(y_true, y_pred)
    
    # R-squared
    if 'r2' in metrics:
        results['r2'] = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    if 'mape' in metrics:
        # Avoid division by zero
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        results['mape'] = mape
    
    # Explained Variance
    if 'explained_variance' in metrics:
        results['explained_variance'] = explained_variance_score(y_true, y_pred)
    
    return results


def plot_actual_vs_predicted(y_true, y_pred, title=None, save_path=None):
    """
    Plot actual vs predicted values
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    title : str or None
        Plot title (default: None)
    save_path : str or None
        Path to save plot (default: None)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Create scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='none')
    
    # Add identity line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.7, label='Identity Line')
    
    # Add regression line
    m, b = np.polyfit(y_true, y_pred, 1)
    ax.plot(np.array(lims), m * np.array(lims) + b, 'r-', label=f'Fit Line (slope={m:.2f})')
    
    # Add metrics to plot
    metrics_text = (
        f"RMSE: {metrics['rmse']:.2f}\n"
        f"MAE: {metrics['mae']:.2f}\n"
        f"R²: {metrics['r2']:.4f}\n"
        f"MAPE: {metrics['mape']:.2f}%"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Set labels and title
    ax.set_xlabel('Actual IC50 (nM)')
    ax.set_ylabel('Predicted IC50 (nM)')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Actual vs Predicted IC50 Values')
    
    # Set equal scales
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_residuals(y_true, y_pred, title=None, save_path=None):
    """
    Plot residuals
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    title : str or None
        Plot title (default: None)
    save_path : str or None
        Path to save plot (default: None)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Scatter plot of residuals vs predicted
    ax1.scatter(y_pred, residuals, alpha=0.5, edgecolors='none')
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted IC50 (nM)')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted Values')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Distribution of residuals
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add metrics to plot
    metrics = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'median': np.median(residuals)
    }
    metrics_text = (
        f"Mean: {metrics['mean']:.2f}\n"
        f"Std Dev: {metrics['std']:.2f}\n"
        f"Median: {metrics['median']:.2f}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('Residual Analysis', fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(model, feature_names, top_n=20, title=None, save_path=None):
    """
    Plot feature importance for tree-based models
    
    Parameters
    ----------
    model : object
        Trained model with feature_importances_ attribute
    feature_names : list
        Names of the features
    top_n : int
        Number of top features to show (default: 20)
    title : str or None
        Plot title (default: None)
    save_path : str or None
        Path to save plot (default: None)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Get indices of top features
    indices = np.argsort(importances)[::-1]
    
    # Select top N features
    top_indices = indices[:min(top_n, len(feature_names))]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_importances, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Importance')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Feature Importance')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), 
                      scoring='neg_mean_squared_error', title=None, save_path=None):
    """
    Plot learning curve for a model
    
    Parameters
    ----------
    model : object
        Model object that implements fit/predict
    X : array-like
        Training data
    y : array-like
        Target values
    cv : int
        Number of cross-validation folds (default: 5)
    train_sizes : array-like
        Array of training set sizes to evaluate (default: np.linspace(0.1, 1.0, 10))
    scoring : str
        Scoring metric for cross-validation (default: 'neg_mean_squared_error')
    title : str or None
        Plot title (default: None)
    save_path : str or None
        Path to save plot (default: None)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, 
        train_sizes=train_sizes, scoring=scoring
    )
    
    # Calculate mean and std for train and test scores
    train_mean = -np.mean(train_scores, axis=1)  # Negative because of neg_mean_squared_error
    train_std = np.std(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1)  # Negative because of neg_mean_squared_error
    test_std = np.std(test_scores, axis=1)
    
    # Convert to RMSE if using MSE
    if scoring == 'neg_mean_squared_error':
        train_mean = np.sqrt(train_mean)
        train_std = train_std / (2 * np.sqrt(train_mean))  # Error propagation
        test_mean = np.sqrt(test_mean)
        test_std = test_std / (2 * np.sqrt(test_mean))  # Error propagation
        y_label = 'Root Mean Squared Error'
    else:
        y_label = scoring.replace('_', ' ').title()
    
    # Plot learning curve
    ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training error')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                   alpha=0.1, color='r')
    
    ax.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation error')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                   alpha=0.1, color='g')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Learning Curve')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_models(models_dict, X, y, cv=5, metrics=None):
    """
    Compare multiple models using cross-validation
    
    Parameters
    ----------
    models_dict : dict
        Dictionary of model name -> model object
    X : array-like
        Feature data
    y : array-like
        Target values
    cv : int
        Number of cross-validation folds (default: 5)
    metrics : list or None
        List of metrics to calculate (default: None)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing performance metrics for each model
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'r2', 'mape']
    
    results = []
    
    for name, model in models_dict.items():
        # Get cross-validated predictions
        y_pred = cross_val_predict(model, X, y, cv=cv)
        
        # Calculate metrics
        model_metrics = calculate_metrics(y, y_pred, metrics)
        model_metrics['Model'] = name
        
        results.append(model_metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns to have Model first
    cols = ['Model'] + [col for col in results_df.columns if col != 'Model']
    results_df = results_df[cols]
    
    return results_df


def plot_model_comparison(comparison_df, metric='rmse', title=None, save_path=None):
    """
    Plot model comparison for a specific metric
    
    Parameters
    ----------
    comparison_df : pandas.DataFrame
        DataFrame from compare_models function
    metric : str
        Metric to plot (default: 'rmse')
    title : str or None
        Plot title (default: None)
    save_path : str or None
        Path to save plot (default: None)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found in comparison DataFrame")
    
    # Sort by the metric
    sorted_df = comparison_df.sort_values(by=metric)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    bar_colors = sns.color_palette("viridis", len(sorted_df))
    bars = ax.bar(sorted_df['Model'], sorted_df[metric], color=bar_colors)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{height:.4f}', ha='center', va='bottom', rotation=0
        )
    
    # Set labels and title
    ax.set_xlabel('Model')
    
    # Format y-label based on metric
    if metric == 'rmse':
        y_label = 'Root Mean Squared Error'
    elif metric == 'mae':
        y_label = 'Mean Absolute Error'
    elif metric == 'r2':
        y_label = 'R² Score'
    elif metric == 'mape':
        y_label = 'Mean Absolute Percentage Error (%)'
    elif metric == 'explained_variance':
        y_label = 'Explained Variance'
    else:
        y_label = metric
    
    ax.set_ylabel(y_label)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Model Comparison - {y_label}')
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define feature names
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Create and train models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    # Make predictions
    rf_preds = models['RandomForest'].predict(X_test)
    
    # Plot actual vs predicted
    fig1 = plot_actual_vs_predicted(y_test, rf_preds, 
                                   title='Random Forest: Actual vs Predicted')
    
    # Plot residuals
    fig2 = plot_residuals(y_test, rf_preds, 
                         title='Random Forest: Residual Analysis')
    
    # Plot feature importance
    fig3 = plot_feature_importance(models['RandomForest'], feature_names, top_n=10,
                                  title='Random Forest: Feature Importance')
    
    # Plot learning curve
    fig4 = plot_learning_curve(models['RandomForest'], X, y, cv=5,
                             title='Random Forest: Learning Curve')
    
    # Compare models
    comparison_df = compare_models(models, X, y, cv=5)
    print(comparison_df)
    
    # Plot model comparison
    fig5 = plot_model_comparison(comparison_df, metric='rmse',
                               title='Model Comparison - RMSE')
    
    plt.show()
