#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for dimension reduction techniques applied to molecular descriptor data

This module provides functions to reduce the dimensionality of molecular descriptor data
using various techniques such as PCA, t-SNE, UMAP, etc.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def preprocess_data(data, numeric_only=True, handle_na='median', scale=True):
    """
    Preprocess data for dimension reduction
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing molecular descriptors
    numeric_only : bool
        Whether to include only numeric columns (default: True)
    handle_na : str or None
        Method to handle NaN values ('median', 'mean', 'drop', None)
    scale : bool
        Whether to standardize the data (default: True)
        
    Returns
    -------
    numpy.ndarray
        Preprocessed data ready for dimension reduction
    list
        Names of the columns included in the preprocessed data
    """
    # Select numeric columns if required
    if numeric_only:
        df = data.select_dtypes(include=[np.number])
    else:
        df = data.copy()
    
    # Store column names
    columns = df.columns.tolist()
    
    # Handle NaN values
    if handle_na == 'median':
        df = df.fillna(df.median())
    elif handle_na == 'mean':
        df = df.fillna(df.mean())
    elif handle_na == 'drop':
        df = df.dropna()
    elif handle_na is not None:
        raise ValueError(f"Unknown NaN handling method: {handle_na}")
    
    # Convert to numpy array
    X = df.values
    
    # Scale the data if required
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, columns


def reduce_dimensions_pca(data, n_components=2, **kwargs):
    """
    Reduce dimensions using Principal Component Analysis (PCA)
    
    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Data containing molecular descriptors
    n_components : int
        Number of components to keep (default: 2)
    **kwargs : dict
        Additional parameters to pass to PCA
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the reduced data
    object
        The fitted PCA object
    """
    # Preprocess data if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        X, columns = preprocess_data(data)
    else:
        X = data
    
    # Perform PCA
    pca = PCA(n_components=n_components, **kwargs)
    X_reduced = pca.fit_transform(X)
    
    # Create DataFrame for the reduced data
    df_reduced = pd.DataFrame(
        X_reduced, 
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data.index if isinstance(data, pd.DataFrame) else None
    )
    
    return df_reduced, pca


def reduce_dimensions_kpca(data, n_components=2, kernel='rbf', **kwargs):
    """
    Reduce dimensions using Kernel Principal Component Analysis (KPCA)
    
    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Data containing molecular descriptors
    n_components : int
        Number of components to keep (default: 2)
    kernel : str
        Kernel type ('linear', 'poly', 'rbf', 'sigmoid', 'cosine')
    **kwargs : dict
        Additional parameters to pass to KernelPCA
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the reduced data
    object
        The fitted KPCA object
    """
    # Preprocess data if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        X, columns = preprocess_data(data)
    else:
        X = data
    
    # Perform Kernel PCA
    kpca = KernelPCA(n_components=n_components, kernel=kernel, **kwargs)
    X_reduced = kpca.fit_transform(X)
    
    # Create DataFrame for the reduced data
    df_reduced = pd.DataFrame(
        X_reduced, 
        columns=[f'KPCA{i+1}' for i in range(n_components)],
        index=data.index if isinstance(data, pd.DataFrame) else None
    )
    
    return df_reduced, kpca


def reduce_dimensions_tsne(data, n_components=2, **kwargs):
    """
    Reduce dimensions using t-Distributed Stochastic Neighbor Embedding (t-SNE)
    
    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Data containing molecular descriptors
    n_components : int
        Number of components to keep (default: 2)
    **kwargs : dict
        Additional parameters to pass to TSNE
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the reduced data
    object
        The fitted TSNE object
    """
    # Preprocess data if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        X, columns = preprocess_data(data)
    else:
        X = data
    
    # Perform t-SNE
    tsne = TSNE(n_components=n_components, **kwargs)
    X_reduced = tsne.fit_transform(X)
    
    # Create DataFrame for the reduced data
    df_reduced = pd.DataFrame(
        X_reduced, 
        columns=[f'TSNE{i+1}' for i in range(n_components)],
        index=data.index if isinstance(data, pd.DataFrame) else None
    )
    
    return df_reduced, tsne


def reduce_dimensions_umap(data, n_components=2, **kwargs):
    """
    Reduce dimensions using Uniform Manifold Approximation and Projection (UMAP)
    
    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Data containing molecular descriptors
    n_components : int
        Number of components to keep (default: 2)
    **kwargs : dict
        Additional parameters to pass to UMAP
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the reduced data
    object
        The fitted UMAP object
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is not available. Please install it with 'pip install umap-learn'")
        
    # Preprocess data if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        X, columns = preprocess_data(data)
    else:
        X = data
    
    # Perform UMAP
    umap = UMAP(n_components=n_components, **kwargs)
    X_reduced = umap.fit_transform(X)
    
    # Create DataFrame for the reduced data
    df_reduced = pd.DataFrame(
        X_reduced, 
        columns=[f'UMAP{i+1}' for i in range(n_components)],
        index=data.index if isinstance(data, pd.DataFrame) else None
    )
    
    return df_reduced, umap


def reduce_dimensions_truncated_svd(data, n_components=2, **kwargs):
    """
    Reduce dimensions using Truncated Singular Value Decomposition (SVD)
    
    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Data containing molecular descriptors
    n_components : int
        Number of components to keep (default: 2)
    **kwargs : dict
        Additional parameters to pass to TruncatedSVD
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the reduced data
    object
        The fitted TruncatedSVD object
    """
    # Preprocess data if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        X, columns = preprocess_data(data)
    else:
        X = data
    
    # Perform Truncated SVD
    svd = TruncatedSVD(n_components=n_components, **kwargs)
    X_reduced = svd.fit_transform(X)
    
    # Create DataFrame for the reduced data
    df_reduced = pd.DataFrame(
        X_reduced, 
        columns=[f'SVD{i+1}' for i in range(n_components)],
        index=data.index if isinstance(data, pd.DataFrame) else None
    )
    
    return df_reduced, svd


def reduce_dimensions(data, method='pca', n_components=2, **kwargs):
    """
    Reduce dimensions using specified method
    
    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Data containing molecular descriptors
    method : str
        Dimension reduction method ('pca', 'kpca', 'tsne', 'umap', 'svd')
    n_components : int
        Number of components to keep (default: 2)
    **kwargs : dict
        Additional parameters to pass to the dimension reduction method
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the reduced data
    object
        The fitted dimension reduction object
    """
    method = method.lower()
    
    if method == 'pca':
        return reduce_dimensions_pca(data, n_components, **kwargs)
    elif method == 'kpca':
        return reduce_dimensions_kpca(data, n_components, **kwargs)
    elif method == 'tsne':
        return reduce_dimensions_tsne(data, n_components, **kwargs)
    elif method == 'umap':
        return reduce_dimensions_umap(data, n_components, **kwargs)
    elif method == 'svd':
        return reduce_dimensions_truncated_svd(data, n_components, **kwargs)
    else:
        raise ValueError(f"Unknown dimension reduction method: {method}")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_blobs
    
    # Create a sample dataset
    X, _ = make_blobs(n_samples=1000, centers=5, cluster_std=1.0, random_state=42)
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
    
    print("Original data shape:", df.shape)
    
    # Reduce dimensions using PCA
    df_pca, pca = reduce_dimensions(df, method='pca', n_components=2)
    
    print("Reduced data shape (PCA):", df_pca.shape)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    
    # Reduce dimensions using t-SNE
    df_tsne, _ = reduce_dimensions(df, method='tsne', n_components=2, perplexity=30)
    
    print("Reduced data shape (t-SNE):", df_tsne.shape)
