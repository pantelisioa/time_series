#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def embed(x, m=1, tau=1):
    """
    Creates an embedding vector using time delay embedding.
    
    Parameters:
    x : array-like
        The input time series.
    m : int, optional
        The embedding dimension (default is 1).
    tau : int, optional
        The time delay (default is 1).
    
    Returns:
    y : ndarray
        The embedded vector of shape (N - (m - 1) * tau, m).
    
    Example:
    N = 300  # length of time series
    x = 0.9 * np.sin(np.arange(1, N + 1) * 2 * np.pi / 70)  # example time series
    y = embed(x, 2, 17)  # embed into 2 dimensions using delay 17
    """
    
    # Ensure x is a 1D or 2D array and convert to a column vector if needed
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    N = x.shape[0] - (m - 1) * tau  # Length of the embedded vector
    if N <= 1:
        raise ValueError("The time series length is too short for the given embedding dimension and delay.")

    # Create the embedded matrix
    if x.shape[1] == 1:  # One-column input time series
        y = np.array([x[i:i + N] for i in range(0, m * tau, tau)]).T
    else:  # Multi-column input time series
        y = np.zeros((N, x.shape[1] * m))
        for i in range(x.shape[1]):
            for j in range(m):
                y[:, i * m + j] = x[j * tau:j * tau + N, i]
    
    return y.T

