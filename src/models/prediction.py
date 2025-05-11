#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for predicting IC50 values for new compounds using trained models

This module provides functions to preprocess new compounds and predict their IC50 values
using trained regression models.
"""

import numpy as np
import pandas as pd


def preprocess_for_prediction(compounds, scaler=None, required_features=None):
    """
    Preprocess compounds for prediction
    
    Parameters
    ----------
    compounds : pandas.DataFrame
        DataFrame containing molecular descriptors for new compounds
    scaler : object or None
        Trained StandardScaler for feature scaling (None for no scaling)
    required_features : list or None
        List of features required by the model (None to use all available)
        
    Returns
    -------
    numpy.ndarray
        Preprocessed features ready for prediction
    """
    # Select only numeric columns
    X = compounds.select_dtypes(include=[np.number])
    
    # Select required features if specified
    if required_features is not None:
        # Check which required features are available
        missing_features = set(required_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Select only the required features in the correct order
        X = X[required_features]
    
    # Fill missing values with median
    X = X.fillna(X.median())
    
    # Scale features if scaler is provided
    if scaler is not None:
        X_preprocessed = scaler.transform(X)
    else:
        X_preprocessed = X.values
    
    return X_preprocessed


def predict_ic50(model, compounds, scaler=None, required_features=None, log_transform=False):
    """
    Predict IC50 values for new compounds
    
    Parameters
    ----------
    model : object
        Trained regression model
    compounds : pandas.DataFrame
        DataFrame containing molecular descriptors for new compounds
    scaler : object or None
        Trained StandardScaler for feature scaling (None for no scaling)
    required_features : list or None
        List of features required by the model (None to use all available)
    log_transform : bool
        Whether to inverse log-transform predictions (default: False)
        
    Returns
    -------
    numpy.ndarray
        Predicted IC50 values for each compound
    """
    # Preprocess compounds
    X_preprocessed = preprocess_for_prediction(compounds, scaler, required_features)
    
    # Make predictions
    predictions = model.predict(X_preprocessed)
    
    # Inverse transform predictions if log-transformed
    if log_transform:
        predictions = np.exp(predictions)
    
    return predictions


def predict_activity(model, compounds, scaler=None, required_features=None, 
                     log_transform=False, activity_threshold=1000):
    """
    Predict activity (active/inactive) based on IC50 values
    
    Parameters
    ----------
    model : object
        Trained regression model
    compounds : pandas.DataFrame
        DataFrame containing molecular descriptors for new compounds
    scaler : object or None
        Trained StandardScaler for feature scaling (None for no scaling)
    required_features : list or None
        List of features required by the model (None to use all available)
    log_transform : bool
        Whether to inverse log-transform predictions (default: False)
    activity_threshold : float
        IC50 threshold for active compounds in nM (default: 1000)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with predicted IC50 values and activity classification
    """
    # Predict IC50 values
    ic50_predictions = predict_ic50(model, compounds, scaler, required_features, log_transform)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Predicted_IC50_nM': ic50_predictions,
        'Activity': ['Active' if ic50 <= activity_threshold else 'Inactive' for ic50 in ic50_predictions]
    })
    
    # Add molecule IDs if available
    if 'molecule_chembl_id' in compounds.columns:
        results['molecule_chembl_id'] = compounds['molecule_chembl_id'].values
    elif 'ID' in compounds.columns:
        results['ID'] = compounds['ID'].values
    
    return results


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    # Create a sample dataset
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
    df['IC50_nM'] = y
    
    # Split data
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    
    # Train a model
    X_train = train_data.drop(columns=['IC50_nM'])
    y_train = train_data['IC50_nM']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predict IC50 for test compounds
    test_compounds = test_data.drop(columns=['IC50_nM'])
    predictions = predict_ic50(model, test_compounds, scaler)
    
    # Classify activity
    activity_results = predict_activity(model, test_compounds, scaler)
    
    print("Sample predictions:")
    print(activity_results.head())
    
    # Calculate accuracy
    true_active = (test_data['IC50_nM'] <= 0).astype(int)  # Assuming 0 is the threshold for this example
    pred_active = (activity_results['Predicted_IC50_nM'] <= 0).astype(int)
    accuracy = (true_active == pred_active).mean()
    
    print(f"\nPrediction accuracy: {accuracy:.4f}")
