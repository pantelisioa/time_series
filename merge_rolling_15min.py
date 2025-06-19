#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def merge_rolling_15min(patient, is_control=False):
    """
    Create rolling 15-minute windows from night-time HRV data and calculate the coefficient of variation (CV).
    
    Args:
        patient: Identifier for the patient or control group.
        is_control: Boolean indicating whether the patient belongs to the control group.
    
    Returns:
        hrv_total_night_night: Merged HRV data as a single array.
        hrv_rolling_15min: List of HRV data for each 15-minute rolling window.
        cv: List of coefficients of variation for each rolling window.
    """
    # Concatenate the night-time HRV data
    hrv_total_night = []
    
    # Select appropriate night indices and data
    if is_control:
        night_indices = night_indices_list_control[patient]
        df_night = df_night_control[patient]
        for idx in night_indices:
            if idx in df_night.index:
                hrv_total_night.extend(df_night.loc[idx])  # Use `.loc` for index-based selection
            else:
                print(f"Warning: Index {idx} not found for patient {patient}. Skipping.")
    else:
        night_indices = night_indices_list_patients[patient]
        df_night = df_night_patients[patient]
        for idx in night_indices:
            if idx in df_night.index:
                hrv_total_night.extend(df_night.loc[idx])  # Use `.loc` for index-based selection
            else:
                print(f"Warning: Index {idx} not found for patient {patient}. Skipping.")
            
            
    # Convert to a NumPy array for further calculations
    hrv_total_night_night = np.array(hrv_total_night, dtype=float)

    # Create rolling 15-minute windows
    hrv_rolling_15min = []  # List to store HRV data for each 15-minute rolling window
    cv = []  # List to store coefficients of variation for each window
    for k in range(len(hrv_total_night_night) - 990):
        data_15min = hrv_total_night_night[k:k+990]
        hrv_rolling_15min.append(data_15min)
        
        # Calculate coefficient of variation (CV)
        mean_value = np.mean(data_15min)
        std_dev = np.std(data_15min)
        cv.append(std_dev / mean_value**3 if mean_value != 0 else np.nan)  # Avoid division by zero

    return hrv_rolling_15min, cv

