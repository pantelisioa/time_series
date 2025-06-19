#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

# Calculate the Kullback-Leibler divergence

def KL(a, b):
    """This function calculates the Kullback-Leibler divergence of the distributions a and b"""
    # Ensure that a and b are numpy arrays
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    # Avoid division by zero or log of zero by using a small constant
    eps = 1e-10
    a = np.where(a == 0, eps, a)
    b = np.where(b == 0, eps, b)

    # Make sure the distributions have the same length
    if len(a) > len(b):
        a = a[:len(b)]
    elif len(b) > len(a):
        b = b[:len(a)]

    # Calculate KL divergence
    kl = sp.stats.entropy(a,b)
    
    return kl

