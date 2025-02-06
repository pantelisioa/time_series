import numpy as np
import scipy as sp

def SampEn(x,m=1,r=0.2,N=len(x)):
    """
    Calculates the sample entropy of the time series x.

    Input:
        x -- Timeseries array
        m -- Embedding Dimension, default m=1
        r -- Tolerance, default r=0.2*std
        N -- Length of the segment of the timeseries that we want to calculate the SampEn, default N=len(x)

    Output:
        sampen -- Sample Entropy
        se -- Standard error estimates

    """
    # define tolerance
    tol = r*np.std(x)

    # create the m-dimensional vector space for x0
    x0 = np.zeros((m,N))
    x1 = np.zeros((m+1,N))

    # Embedding in the m dimensionality
    for i in range(m):
        x0[i,i:] = x[i:]
        x0[i] = np.roll(x[i],N-i)
        x0 = x0.T

    # Do the same for the m+1-dimensional vector space for x1
    for j in range(m+1):
        x1[j,j:] = x[j:]
        x1[j] = np.roll(x[j],N-j)
        x1 = x1.T