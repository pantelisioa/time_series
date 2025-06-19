#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def adf_test(timeseries):
    """
    Augmented Dickey Fuller (“ADF”) test to check for stationarity.
    
    Input: 
        timeseries -- time series to be tested if is stationary
    Output:
        p-Value -- Null Hypothesis: The series has a unit root.

                    Alternate Hypothesis: The series has no unit root.

    If the null hypothesis in failed to be rejected, this test may provide evidence that the series is non-stationary.
    """
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
    return dfoutput["p-value"]

