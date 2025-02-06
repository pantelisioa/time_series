#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def euklidean(x,y):
    "Calculates the euklidian distance of two vectors x and y"
    diff = []
    for i in range(len(x)):
        diff.append((x[i]-y[i])**2)
    dist = np.sqrt(sum(diff))
    return dist

