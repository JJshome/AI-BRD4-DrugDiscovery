#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for machine learning regression models to predict IC50 values

This module provides functions to train and evaluate regression models for predicting
IC50 values of BRD4 inhibitors based on molecular descriptors.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def prepare_data(data, target_column='IC50_nM', test_size=0.2, random_state=42, scale=True):
    """
    Prepare data for model training and evaluation
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing molecular descriptors and IC50 values
    target_column : str
        Name of the column containing IC50 values (default: 'IC50_nM')
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    scale : bool
        Whether to standardize features (default: True)
        
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, scaler)
    """
    # Check if target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Remove non-numeric columns (if any)
    X = X.select_dtypes(include=[np.number])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale features if required
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None
    
    return X_train, X_test, y_train, y_test, scaler


def train_random_forest(X_train, y_train, **kwargs):
    """
    Train a Random Forest regressor
    
    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    **kwargs : dict
        Additional parameters to pass to RandomForestRegressor
        
    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model
    """
    # Set default parameters if not provided
    params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }
    params.update(kwargs)
    
    # Create and train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    return model


def train_gradient_boosting(X_train, y_train, **kwargs):
    """
    Train a Gradient Boosting regressor
    
    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    **kwargs : dict
        Additional parameters to pass to GradientBoostingRegressor
        
    Returns
    -------
    sklearn.ensemble.GradientBoostingRegressor
        Trained Gradient Boosting model
    """
    # Set default parameters if not provided
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }
    params.update(kwargs)
    
    # Create and train model
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    
    return model


def train_elastic_net(X_train, y_train, **kwargs):
    """
    Train an Elastic Net regressor
    
    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    **kwargs : dict
        Additional parameters to pass to ElasticNet
        
    Returns
    -------
    sklearn.linear_model.ElasticNet
        Trained Elastic Net model
    """
    # Set default parameters if not provided
    params = {
        'alpha': 1.0,
        'l1_ratio': 0.5,
        'max_iter': 1000,
        'random_state': 42
    }
    params.update(kwargs)
    
    # Create and train model
    model = ElasticNet(**params)
    model.fit(X_train, y_train)
    
    return model


def train_svr(X_train, y_train, **kwargs):
    """
    Train a Support Vector Regression model
    
    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    **kwargs : dict
        Additional parameters to pass to SVR
        
    Returns
    -------
    sklearn.svm.SVR
        Trained SVR model
    """
    # Set default parameters if not provided
    params = {
        'kernel': 'rbf',
        'C': 1.0,
        'epsilon': 0.1,
        'gamma': 'scale'
    }
    params.update(kwargs)
    
    # Create and train model
    model = SVR(**params)
    model.fit(X_train, y_train)
    
    return model


def train_mlp(X_train, y_train, **kwargs):
    """
    Train a Multi-Layer Perceptron regressor
    
    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    **kwargs : dict
        Additional parameters to pass to MLPRegressor
        
    Returns
    -------
    sklearn.neural_network.MLPRegressor
        Trained MLP model
    """
    # Set default parameters if not provided
    params = {
        'hidden_layer_sizes': (100,),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'max_iter': 200,
        'random_state': 42
    }
    params.update(kwargs)
    
    # Create and train model
    model = MLPRegressor(**params)
    model.fit(X_train, y_train)
    
    return model


def train_model(X_train, y_train, model_type='random_forest', **kwargs):
    """
    Train a regression model using the specified algorithm
    
    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    model_type : str
        Type of model to train ('random_forest', 'gradient_boosting', 'elastic_net', 'svr', 'mlp')
    **kwargs : dict
        Additional parameters to pass to the model
        
    Returns
    -------
    object
        Trained regression model
    """
    model_type = model_type.lower()
    
    if model_type == 'random_forest':
        return train_random_forest(X_train, y_train, **kwargs)
    elif model_type == 'gradient_boosting':
        return train_gradient_boosting(X_train, y_train, **kwargs)
    elif model_type == 'elastic_net':
        return train_elastic_net(X_train, y_train, **kwargs)
    elif model_type == 'svr':
        return train_svr(X_train, y_train, **kwargs)
    elif model_type == 'mlp':
        return train_mlp(X_train, y_train, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(model, X_test, y_test, metrics=None):
    """
    Evaluate a trained model on test data
    
    Parameters
    ----------
    model : object
        Trained regression model
    X_test : array-like
        Test features
    y_test : array-like
        Test targets
    metrics : list or None
        List of metrics to compute (None for all)
        Options: 'rmse', 'mae', 'r2', 'mape'
        
    Returns
    -------
    dict
        Dictionary containing computed metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Initialize results
    results = {}
    
    # Compute requested metrics
    if metrics is None or 'rmse' in metrics:
        results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
    
    if metrics is None or 'mae' in metrics:
        results['mae'] = mean_absolute_error(y_test, y_pred)
    
    if metrics is None or 'r2' in metrics:
        results['r2'] = r2_score(y_test, y_pred)
    
    if metrics is None or 'mape' in metrics:
        # Mean Absolute Percentage Error (avoiding division by zero)
        mask = y_test != 0
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        results['mape'] = mape
    
    return results


def optimize_hyperparameters(X_train, y_train, model_type='random_forest', param_grid=None, cv=5):
    """
    Optimize hyperparameters for a regression model using grid search
    
    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    model_type : str
        Type of model to train ('random_forest', 'gradient_boosting', 'elastic_net', 'svr', 'mlp')
    param_grid : dict or None
        Grid of hyperparameters to search (None for default grid)
    cv : int
        Number of cross-validation folds (default: 5)
        
    Returns
    -------
    dict
        Best hyperparameters found
    float
        Best cross-validation score
    """
    model_type = model_type.lower()
    
    # Define default parameter grids if not provided
    if param_grid is None:
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        elif model_type == 'elastic_net':
            param_grid = {
                'alpha': [0.1, 0.5, 1.0, 2.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9]
            }
        elif model_type == 'svr':
            param_grid = {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1.0, 10.0],
                'epsilon': [0.01, 0.1, 0.2],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        elif model_type == 'mlp':
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'solver': ['adam', 'lbfgs']
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Create base model
    if model_type == 'random_forest':
        base_model = RandomForestRegressor(random_state=42)
    elif model_type == 'gradient_boosting':
        base_model = GradientBoostingRegressor(random_state=42)
    elif model_type == 'elastic_net':
        base_model = ElasticNet(random_state=42)
    elif model_type == 'svr':
        base_model = SVR()
    elif model_type == 'mlp':
        base_model = MLPRegressor(random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Perform grid search
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=cv, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = np.sqrt(-grid_search.best_score_)  # RMSE from negative MSE
    
    return best_params, best_score


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    
    # Create a sample dataset
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
    df['IC50_nM'] = y
    
    # Prepare data
    X_train, X_test, y_train, y_test, _ = prepare_data(df)
    
    print("Training Random Forest model...")
    rf_model = train_model(X_train, y_train, model_type='random_forest')
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print("Random Forest metrics:", rf_metrics)
    
    print("\nTraining Gradient Boosting model...")
    gb_model = train_model(X_train, y_train, model_type='gradient_boosting')
    gb_metrics = evaluate_model(gb_model, X_test, y_test)
    print("Gradient Boosting metrics:", gb_metrics)
