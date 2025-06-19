#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import adf_test

def stat_epoch(ts,window):
    """
    Calculates the stationary epoch of a HRV.
    
    Input:
        ts -- HRV time series in ms
        window -- Stationary epoch of length of the specified window in minutes
    Output:
        stat_epoch --  the segment where the HRV time series is supposed to be stationary.
    """
    
    w = window*60*1000 #25min in ms
    total_duration = np.sum(ts) # total duration of the time series in ms
    num_of_windows = np.round(total_duration/w) # number of [window]min windows that do not coincide
    num_of_indices = int(len(ts)/num_of_windows) # approx number of indices that make approx [window]min
    
    # make a list of the CV of the 20min windows
    cv_windows = [] 
    for i in range(len(ts)-num_of_indices):
        cv_windows.append(np.std(ts[i:i+num_of_indices])/np.mean(ts[i:i+num_of_indices])**5)
        
    # check where the cv is minimal
    minimal_index = np.where(cv_windows == np.min(cv_windows))[0]
    stat_epoch = ts[minimal_index[0]:minimal_index[0]+num_of_indices]
    
    p_Value = adf_test(stat_epoch)
    
    if p_Value <0.05:
        print("ADF-Test shows that the stat_epoch seems stationary")
    return stat_epoch, p_Value

