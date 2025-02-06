#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def embed(x, m=1, tau=1):
    """
    Creates a time delay embedding of a time series.
    
    Parameters:
        x : array-like
            The input time series.
        m : int, optional
            The embedding dimension (default is 1).
        tau : int, optional
            The time delay (default is 1).
    
    Returns:
        y : ndarray
            An array of shape (m, L) where L = len(x) - (m-1)*tau. Each column is an embedded vector.
    
    Example:
        N = 300
        x = 0.9 * np.sin(np.arange(1, N + 1) * 2 * np.pi / 70)
        y = embed(x, 2, 17)  # Embedding into 2 dimensions with delay 17.
    """
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    N = x.shape[0] - (m - 1) * tau
    if N <= 0:
        raise ValueError("The time series length is too short for the given embedding dimension and delay.")

    # For a one-dimensional series, create a matrix where each row is a delayed version.
    if x.shape[1] == 1:
        y = np.column_stack([x[i: i + N] for i in range(0, m * tau, tau)])
    else:
        # For multivariate time series.
        y = np.zeros((N, x.shape[1] * m))
        for i in range(x.shape[1]):
            for j in range(m):
                y[:, i * m + j] = x[j * tau: j * tau + N, i]
    return y.T  # Return shape (m, L)

