#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

# Function to calculate the HRV out of the peaks
def HRV(data,samplingrate):
    """
    Input:
    data: list of peak indices
    samplingrate: sample rate
    
    Output:
    hrv: 1D time series array
    
    """
    hrv = []
    for l in range(len(data)-1):
        hrv.append(abs(data[l]-data[l+1])*(1/samplingrate))
    return np.array(hrv)

