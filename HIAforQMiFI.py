import pandas as pd

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