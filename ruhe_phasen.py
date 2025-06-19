#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

# Now pick out the parts where all three directions do not have Std_dev more than 0.3

def ruhe_phasen(variance_x,variance_y,variance_z,threshold):
    'This functions gives me back the indices where the variance in all directions do not exceed the threshold'
    indices_x = np.where(variance_x < threshold)
    indices_y = np.where(variance_y < threshold)
    indices_z = np.where(variance_z < threshold)
    
    # get the common indices
    common_indices = np.intersect1d(np.intersect1d(indices_x, indices_y), indices_z)
    return common_indices

