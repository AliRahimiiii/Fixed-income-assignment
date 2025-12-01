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
    def recursive_forecast(model, last_known_value, h):
        """
        Performs recursive forecasting using a fitted AR model.

        Parameters
        ----------
        model : sklearn.linear_model.LinearRegression
            The fitted linear regression model for forecasting.
        last_known_value : float
            The last known value to start the forecast from.
        h : int
            The forecast horizon (number of steps to forecast).
        Returns
        -------
        float
            The forecasted value after h steps.
        """
        # Ensure input is 2D (1, 1) for sklearn
        current_val = last_known_value.reshape(1, -1)
        
        for _ in range(h):
            current_val = model.predict(current_val)
            
        return current_val[0, 0].item()




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
    B_slice = B[:, :, :i]                                                                                           # shape (120, 3, i)
    y_slice = y[:, :i]                                                                                             # shape (120, i)

    # initialize beta parameters dictionary
    beta = {
        1 : [],
        2 : [],
        3: []
    }

    # estimate beta parameters for each time t in the sample up to i
    for t in range(i):    
        model = LinearRegression(fit_intercept=False)
        model.fit(B_slice[:,:,t], y_slice[:,t])
        beta[1].append(model.coef_[0].item())
        beta[2].append(model.coef_[1].item())
        beta[3].append(model.coef_[2].item())

    # reshape beta parameters
    beta = {k: np.array(v).reshape(-1, 1) for k, v in beta.items()}


    # estimate AR(1) for each beta parameter
    ar_beta1 = LinearRegression(fit_intercept=True)
    ar_beta1.fit(beta[1][0:i-1], beta[1][1:i])
    ar_beta2 = LinearRegression(fit_intercept=True)
    ar_beta2.fit(beta[2][0:i-1], beta[2][1:i])
    ar_beta3 = LinearRegression(fit_intercept=True)
    ar_beta3.fit(beta[3][0:i-1], beta[3][1:i])

    # forecast beta parameters h steps ahead
    beta1_forecast = recursive_forecast(ar_beta1, beta[1][i-1], h)
    beta2_forecast = recursive_forecast(ar_beta2, beta[2][i-1], h)
    beta3_forecast = recursive_forecast(ar_beta3, beta[3][i-1], h)

    # calculate the forecasted yield for maturity j at time i+h
    tau_hat = np.stack([beta1_forecast, beta2_forecast, beta3_forecast]) @ B[j-1, :, i+h].reshape(-1, 1)

    return tau_hat
