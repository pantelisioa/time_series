#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy as sp

# Now calculate the frequency parameters for all the 5min windows

def FrequencyParameters(hrv_data, fs=1.0):
    # Calculate Power Spectral Density using Welch's method
    
    # Define the frequency ranges:
    ULF_range = (0, 0.0033)
    VLF_range = (0.0033, 0.04)
    LF_range = (0.04, 0.15)
    HF_range = (0.15, 0.4)
    VHF_range = (0.4, 0.5)

    # Initialize lists to hold the power values
    total_power = []
    ULF_power = []
    VLF_power = []
    LF_power = []
    HF_power = []
    VHF_power = []

    for w in range(len(hrv_data)):
        # Compute the power spectral density (PSD) for each window
        nperseg=min(len(hrv_data[w]),256)
        freq, PSD = sp.signal.welch(hrv_data[w], fs=fs,nperseg=nperseg)

        # Integrate over the entire frequency range to calculate total power
        total_power.append(np.trapezoid(PSD, freq))

        # Calculate power in each frequency band
        ULF_power.append(np.trapezoid(PSD[(freq >= ULF_range[0]) & (freq <= ULF_range[1])], 
                                  freq[(freq >= ULF_range[0]) & (freq <= ULF_range[1])]))

        VLF_power.append(np.trapezoid(PSD[(freq >= VLF_range[0]) & (freq <= VLF_range[1])], 
                                  freq[(freq >= VLF_range[0]) & (freq <= VLF_range[1])]))

        LF_power.append(np.trapezoid(PSD[(freq >= LF_range[0]) & (freq <= LF_range[1])], 
                                 freq[(freq >= LF_range[0]) & (freq <= LF_range[1])]))

        HF_power.append(np.trapezoid(PSD[(freq >= HF_range[0]) & (freq <= HF_range[1])], 
                                 freq[(freq >= HF_range[0]) & (freq <= HF_range[1])]))

        VHF_power.append(np.trapezoid(PSD[(freq >= VHF_range[0]) & (freq <= VHF_range[1])], 
                                  freq[(freq >= VHF_range[0]) & (freq <= VHF_range[1])]))

    # Create a DataFrame to store the frequency domain parameters
    params = ['TP', 'ULF', 'VLF', 'LF', 'HF', 'VHF', 'LF/HF', 'LF/TP', 'HF/TP']
    frequency_parameters_windows = pd.DataFrame(columns=params)

    # Populate the DataFrame with the calculated values
    for ww in range(len(hrv_data)):
        frequency_parameters_windows.loc[ww, 'TP'] = total_power[ww]
        frequency_parameters_windows.loc[ww, 'ULF'] = ULF_power[ww]
        frequency_parameters_windows.loc[ww, 'VLF'] = VLF_power[ww]
        frequency_parameters_windows.loc[ww, 'LF'] = LF_power[ww]
        frequency_parameters_windows.loc[ww, 'HF'] = HF_power[ww]
        frequency_parameters_windows.loc[ww, 'VHF'] = VHF_power[ww]
        frequency_parameters_windows.loc[ww, 'LF/HF'] = LF_power[ww] / HF_power[ww] if HF_power[ww] != 0 else np.nan
        frequency_parameters_windows.loc[ww, 'LF/TP'] = LF_power[ww] / total_power[ww] if total_power[ww] != 0 else np.nan
        frequency_parameters_windows.loc[ww, 'HF/TP'] = HF_power[ww] / total_power[ww] if total_power[ww] != 0 else np.nan

    return frequency_parameters_windows

