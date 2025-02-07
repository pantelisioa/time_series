import numpy as np
from scipy.spatial.distance import pdist, squareform

def SampEn(x, m=2, r=0.2, metric='chebyshev'):
    """
    Calculates the sample entropy of the time series x.

    Inputs:
        x -- Time series as a NumPy array.
        m -- Embedding dimension (default m=2).
        r -- Tolerance (default r=0.2 times the standard deviation of x).
        metric -- Distance metric to use (default is 'chebyshev').

    Outputs:
        sampen -- Sample entropy value.
        se -- Standard error estimate.
    """
    # Calculate tolerance as r times the standard deviation
    tol = r * np.std(x)

    # Length of time series
    N = len(x)

    # Create embedded vectors for dimension m and m+1
    x_embedded_m = np.array([x[i:N - m + i] for i in range(m)]).T
    x_embedded_m1 = np.array([x[i:N - m - 1 + i] for i in range(m + 1)]).T

    # Compute pairwise distances
    dist_m = squareform(pdist(x_embedded_m, metric))
    dist_m1 = squareform(pdist(x_embedded_m1, metric))

    # Count pairs that meet the tolerance condition
    B = np.sum(np.sum(dist_m <= tol, axis=1) - 1)  # Subtract 1 to exclude self-pairing
    A = np.sum(np.sum(dist_m1 <= tol, axis=1) - 1)

    # Avoid division by zero
    if B == 0:
        return np.inf, np.nan  # Infinite entropy if B is 0

    # Sample entropy calculation
    sampen = -np.log(A / B)

    # Standard error calculation, taken from the Matlab code of Moorman and Lake (s. Physionet)
    p = A / B
    var_p = (p * (1 - p)) / B
    se = np.sqrt(var_p) / p

    return sampen, se


# Example usage
time_series = np.sin(np.linspace(0, 8 * np.pi, 300)) + 0.1 * np.random.randn(300)
sampen, se = SampEn(time_series, m=2, r=0.2)

print("Sample Entropy:", sampen)
print("Standard Error:", se)
