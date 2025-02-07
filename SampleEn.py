import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist

def SampEn(x,m=1,r=0.2,N=len(x), metric='chebyshev'):
    """
    Calculates the sample entropy of the time series x. Chebyshev is used as the distance metric.

    Input:
        x -- Timeseries array
        m -- Embedding Dimension, default m=1
        r -- Tolerance, default r=0.2*std
        N -- Length of the segment of the timeseries that we want to calculate the SampEn, default N=len(x)
        metric -- Denotes the distance metric, default metric = 'chebyshev' (see scipy.spatial.distance.pdist)

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

    dist0 = pdist(x0, metric)
    dist1 = pdist(x1, metric)

    # Get the number of distances that fullfill the tolerance criterion
    crit0 = np.where(dist0<=tol)[0]
    crit1 = np.where(dist1<=tol)[0]

    # Set the A and B according to the SampEn definition
    A = len(crit1)
    B = len(crit0)

    p = A/B
    # Calculate the SampEn
    sampen = -np.log(p) # RETURN

    # Now lets calculate the standard error estimates
    # This SEE is taken from the MATLAB code of Moorman and Lake
    # https://www.physionet.org/content/sampen/1.0.0/matlab/#files-panel

    var_p = (p*(1-p))/B # Variance of p
    std_p = np.sqrt(var_p)
    se = std_p/p # RETURN

    return sampen, se
