#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy as sp
import pandas as pd

# Calculate different HRV time parameters

def meanNN(hrv_data): # mean of hrv
    return np.mean(hrv_data)

def sdNN(hrv_data): # Std of hrv
    return np.std(hrv_data)

def RMSSD(hrv_data): # The square root of the mean of the squared successive differences between adjacent RR intervals
    ssd2 = []
    for q in range(len(hrv_data)-1):
        ssd2.append((hrv_data[q]-hrv_data[q+1])**2)
        rmssd = np.sqrt(np.mean(ssd2))
    return rmssd

def SDSD(hrv_data): # The standard deviation of the successive differences between RR intervals.
    sd= []
    for q in range(len(hrv_data)-1):
        sd.append((hrv_data[q]-hrv_data[q+1]))
        sdsd = np.std(sd)
    return sdsd

def CVNN(hrv_data): # The standard deviation of the RR intervals (SDNN) divided by the mean of the RR intervals (MeanNN)
    return sdNN(hrv_data)/meanNN(hrv_data)
    
def CVSD(hrv_data): # The root mean square of successive differences (RMSSD) divided by the mean of the RR intervals (MeanNN).
    return RMSSD(hrv_data)/meanNN(hrv_data)

def medianNN(hrv_data): # The median of the RR intervals.
    return np.median(hrv_data)

def madNN(hrv_data): # The median absolute deviation of the RR intervals.
    return sp.stats.median_abs_deviation(hrv_data)

def MCVNN(hrv_data): # The median absolute deviation of the RR intervals (MadNN) divided by the median of the RR intervals (MedianNN).
    return madNN(hrv_data)/medianNN(hrv_data)

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
def Q1(hrv_data):
    return np.percentile(hrv_data,25)
def Q3(hrv_data):
    return np.percentile(hrv_data,75)
def IQRNN(hrv_data): # The interquartile range (IQR) of the RR intervals.
    return Q3(hrv_data)-Q1(hrv_data)

def SDRMSSD(hrv_data): # SDNN / RMSSD, a time-domain equivalent for the low Frequency-to-High Frequency (LF/HF) Ratio (Sollers et al., 2007).
    return sdNN(hrv_data)/RMSSD(hrv_data)

def prc20NN(hrv_data): # Prc20NN: The 20th percentile of the RR intervals (Han, 2017; Hovsepian, 2015).
    return np.percentile(hrv_data,20)

def prc80NN(hrv_data): # Prc80NN: The 80th percentile of the RR intervals (Han, 2017; Hovsepian, 2015).
    return np.percentile(hrv_data,80)

def pNN50(hrv_data):
    """
    Calculate pNN50: The percentage of absolute differences in successive 
    RR intervals greater than 50 ms.

    Parameters:
    hrv_data (list or ndarray): Array of RR intervals.

    Returns:
    float: pNN50 value.
    """
    if len(hrv_data) < 2:
        return 0.0  # Not enough data points to calculate differences
    
    # Calculate the absolute differences between successive RR intervals
    absdiff = np.abs(np.diff(hrv_data))
    
    # Count the number of differences greater than 50 ms
    count = np.sum(absdiff > 50)
    
    # Calculate the percentage
    pnn50_value = (count / len(absdiff)) * 100
    
    return pnn50_value

def pNN20(hrv_data):
    """
    Calculate pNN20: The percentage of absolute differences in successive 
    RR intervals greater than 20 ms.

    Parameters:
    hrv_data (list or ndarray): Array of RR intervals.

    Returns:
    float: pNN20 value.
    """
    if len(hrv_data) < 2:
        return 0.0  # Not enough data points to calculate differences
    
    # Calculate the absolute differences between successive RR intervals
    absdiff = np.abs(np.diff(hrv_data))
    
    # Count the number of differences greater than 20 ms
    count = np.sum(absdiff > 20)
    
    # Calculate the percentage
    pnn20_value = (count / len(absdiff)) * 100
    
    return pnn20_value

def minRR(hrv_data): # the minimum value
    return min(hrv_data)
def maxRR(hrv_data): # the maximum value
    return max(hrv_data)

# Create a Data Frame to add all the time parameter values for each 5min window

def TimeParameters(hrv_data):
    " hrv_data is a list of all the data splitted in 5min windows"
    
    columns = ['meanNN','sdNN','RMSSD','SDSD','CVNN','CVSD','medianNN','madNN','MCVNN','Q1','Q3','IQRNN','SDRMSSD','prc20NN','prc80NN','pNN50','pNN20','minRR','maxRR']
    time_parameters = pd.DataFrame(columns=columns)
    # make a for-loop to add the values
    for tt in range(len(hrv_data)):
        #time_parameters.at[len(time_parameters), 'num. og 5min window'] = count
        time_parameters.loc[tt, 'meanNN'] = meanNN(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'sdNN'] = sdNN(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'RMSSD'] = RMSSD(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'SDSD'] = SDSD(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'CVNN'] = CVNN(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'CVSD'] = CVSD(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'medianNN'] = medianNN(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'madNN'] = madNN(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'MCVNN'] = MCVNN(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'Q1'] = Q1(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'Q3'] = Q3(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'IQRNN'] = IQRNN(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'SDRMSSD'] = SDRMSSD(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'prc20NN'] = prc20NN(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'prc80NN'] = prc80NN(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'pNN50'] = pNN50(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'pNN20'] = pNN20(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'minRR'] = minRR(hrv_data[tt]*1000)
        time_parameters.loc[tt, 'maxRR'] = maxRR(hrv_data[tt]*1000)
        
    return time_parameters

