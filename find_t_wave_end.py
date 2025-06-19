#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import neurokit2 as nk

def find_t_wave_end(ecg, ind_R, fs=1000, total_duration=1.0):
    """
    Finds the T wave end in an ECG signal based on geometric distance.
    
    Parameters:
    - ecg: NumPy array representing ECG signal
    - ind_R: Index of the R peak
    - fs: Sampling rate of the ECG (default: 1000 Hz)
    - total_duration: Total duration of the ECG signal in seconds (default: 1 second)
    
    Returns:
    - T_end: Index of the T wave end
    """
    # Scaling factor for new sample rates
    scale_factor = fs / 1000  # Ratio compared to 1000 Hz

    # Define search range, adjusted for the new sampling rate
    beg_st = ind_R + int(60 * scale_factor)  # 60 ms after R-peak
    after_end_t = int((total_duration * fs) - (50 * scale_factor))  # 50 ms before end

    # Ensure the indices are within bounds
    beg_st = min(beg_st, len(ecg) - 1)
    after_end_t = min(after_end_t, len(ecg) - 1)

    # Find T wave peak (max absolute value in the range)
    ind_T = np.argmax(np.abs(ecg[beg_st:after_end_t])) + beg_st

    # Initialize distance array
    distance = np.zeros_like(ecg)

    # Define the reference points for line distance calculation
    x1, y1 = ind_T, ecg[ind_T]
    x2, y2 = after_end_t, ecg[after_end_t]

    # Compute denominator (line segment length)
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    # Compute perpendicular distance for each point from the reference line
    for i in range(ind_T, after_end_t):
        x0, y0 = i, ecg[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        distance[i] = numerator / denominator

    # Find the index with the maximum perpendicular distance (T wave end)
    end_T1 = np.argmax(distance[ind_T:after_end_t])
    T_end = ind_T + end_T1

    return T_end

