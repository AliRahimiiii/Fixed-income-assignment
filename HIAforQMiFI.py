import pandas as pd
from sklearn.linear_model import LinearRegression


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
    
    # Placeholder for the actual implementation of Nelson-Siegel forecasting
    return data  # Placeholder for the actual implementation
