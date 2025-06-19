#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.spatial.distance import pdist, squareform, cKDTree

def standard_fnn(time_series, m, tau, threshold=10):
    """
    Calculates the percentage of false nearest neighbors (FNN) for a time series
    using standard FNN algorithm. For each point in the m-dimensional embedding,
    it finds its nearest neighbor and then checks the relative increase in distance
    when the embedding is increased to m+1 dimensions.
    
    Parameters:
        time_series : 1D array-like
            The input time series.
        m : int
            The embedding dimension.
        tau : int
            The time delay.
        threshold : float, optional
            The tolerance threshold for relative distance increase.
    
    Returns:
        fnn_percentage : float
            The percentage of false nearest neighbors.
    """
    time_series = np.asarray(time_series)
    N = len(time_series)
    
    # Embed in dimension m.
    X_m = embed(time_series, m, tau)  # shape (m, L_m)
    L_m = X_m.shape[1]
    
    # For a valid comparison, we need to embed in m+1 dimensions.
    X_m1 = embed(time_series, m+1, tau)  # shape (m+1, L_m1)
    L_m1 = X_m1.shape[1]
    
    # We will only compare points where the (m+1)-embedding is available.
    # In other words, we iterate over i=0,...,L_m1-1.
    false_count = 0
    total_count = 0

    # Build a KD-tree for the m-dimensional embedding to find nearest neighbors quickly.
    tree = cKDTree(X_m[:, :L_m1].T)  # each point is a row; shape: (L_m1, m)
    
    # For each point in the m-dimensional embedding (only up to L_m1 points)
    for i in range(L_m1):
        # Query the nearest neighbor (excluding the point itself)
        dist, idx = tree.query(X_m[:, i].T, k=2)
        d_m = dist[1]  # nearest neighbor distance in m-dimensions
        j = idx[1]    # index of nearest neighbor
        
        # In the m+1-dimensional embedding, retrieve the additional coordinate for i and j.
        # Here the additional coordinate is the last row of X_m1.
        # Note: X_m1 is of shape (m+1, L_m1)
        extra_i = X_m1[m, i]
        extra_j = X_m1[m, j]
        
        # Compute the extra distance (absolute difference) in the (m+1)-th coordinate.
        d_extra = np.abs(extra_i - extra_j)
        
        # If the distance in m-dimensions is zero, handle carefully.
        if d_m == 0:
            # If the extra difference is nonzero, consider it as false.
            if d_extra != 0:
                false_count += 1
        else:
            # If the relative increase exceeds the threshold, count it as false.
            if (d_extra / d_m) > threshold:
                false_count += 1
        
        total_count += 1

    fnn_percentage = 100 * false_count / total_count if total_count > 0 else 0
    return fnn_percentage

