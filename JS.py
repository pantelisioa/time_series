#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy as sp


def JS(a,b):
    "This function calculates the Jensen-Shannon divergence of the distributions a and b"
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
    
    # Calculate the mixure distribution m of a and b
    m = 0.5 * (a+b)
    
    # Calculate JS divergence
    js = 0.5*(sp.stats.entropy(a,m) + sp.stats.entropy(b,m))
    
    return js

