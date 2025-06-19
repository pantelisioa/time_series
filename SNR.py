#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

# Compute the SNR avoiding calculating logarithm with zero or dividing by zero

# Avoid division by zero by setting noise_estimate and ecg to a very small number where it's zero
def SNR(signal,noise):
    # Avoid division by zero by setting noise_estimate and ecg to a very small number where it's zero
    noise = np.where(noise==0,1e-10,noise)
    signal = np.where(signal==0,1e-10,signal)
    return 20 * np.log10(np.abs(signal / noise)) # Compute the SNR using vectorized operations

