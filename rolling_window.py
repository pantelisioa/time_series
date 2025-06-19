#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

def rolling_window(series, window_size):
    """
    Calculate the rolling mean and variance for a given series.
    
    Parameters:
    - series: A pandas Series representing the data
    - window_size: The size of the rolling window
    
    Returns:
    - A DataFrame containing the rolling mean and variance
    """
    rolling_mean = series.rolling(window=window_size).mean()
    rolling_variance = series.rolling(window=window_size).var()
    
    # Create a DataFrame with mean and variance as columns
    df = pd.DataFrame({'mean': rolling_mean, 'variance': rolling_variance})
    
    return df

