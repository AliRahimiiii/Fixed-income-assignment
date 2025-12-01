import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


def getTimeSeriesMean(j):
    """
    Loads the data and calculates the time series mean for a specific maturity.

    Parameters
    ----------
    j : int
        The index of the maturity to average (ranges from 1 to 120).

    Returns
    -------
    float
        The historical average (time series mean) of the specified maturity.
    """
    data = pd.read_excel('LW_monthly_1972-2024.xlsx')           # Load the data
    return data.iloc[:, j + 1].mean().item()                    # Calculate and return the mean                   


def getNelsonSiegelForecast(i, h, j):
    """
    returns a yield curve forecast using the Nelson-Siegel model for a specific maturity j at time i+h using parameters estimated up to time i.

    Parameters
    ----------
    i : int
        The time index up to which the subsample runs. (ranges from 12 to 612)
    h : int
        The forecast horizon.  (ranges from 1 to 24)   
    j : int
        The index of the maturity to forecast. (ranges from 1 to 120)
    Returns
    ----------
    float
        The Nelson-Siegel forecast for the specified maturity at time i+h.
    """

    data = pd.read_excel('LW_monthly_1972-2024.xlsx')           # Load the data
    data = data.iloc[:, 2:]                                     # Remove year and month columns                  
    data.index = range(1, len(data)+1)                          # Reset index to start from 1
    
    # make the B(lambda) matrix and y vector
    lamba = 0.0609
    B = np.stack([np.ones_like(data.T),
                   (1 - np.exp(-lamba * data.T))/(lamba * data.T),
                     (1 - np.exp(-lamba * data.T))/(lamba * data.T) - np.exp(-lamba * data.T)], axis=1)       # shape (120, 3, 636)
    
    y = np.asarray(data.T)                                                                                    # shape (120, 636)

    # estimate the beta parameters using data up to time i
    B = B[:, :, :i]                                                                                           # shape (120, 3, i)
    y = y[:, :i]                                                                                             # shape (120, i)

    model = LinearRegression(fit_intercept=False)
    model.fit(B, y)

    # estimate AR(1) for each beta parameter
    

    return data  # Placeholder for the actual implementation
