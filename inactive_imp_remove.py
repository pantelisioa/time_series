#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

# now locate the respective ecg positions to remove using the time in seconds and take only integer values
def inactive_imp_remove(ecg,samplerate, values_to_remove):
    mask = np.array([int(index * 1/samplerate) not in values_to_remove for index in range(len(ecg))])
    filtered_vector = ecg[mask]
    return filtered_vector

