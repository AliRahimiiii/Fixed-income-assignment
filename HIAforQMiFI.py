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
    returns a yield curve forecast using the Nelson-Siegel model for a specific maturity j at time i+h
    using parameters estimated up to time i.

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
    tau = np.arange(1, 121)
    B = np.stack([
        np.ones_like(tau),
        (1 - np.exp(-lamba * tau)) / (lamba * tau),
        (1 - np.exp(-lamba * tau)) / (lamba * tau) - np.exp(-lamba * tau)
    ], axis=1)  # shape (120, 3, 636)

    y = np.asarray(data.T)  # shape (120, 636)

    # estimate the beta parameters using data up to time i
    y_slice = y[:, :i]  # shape (120, i)

    # initialize beta parameters dictionary
    beta = {}

    # estimate beta parameters for each time t in the sample up to i
    model = LinearRegression(fit_intercept=False)
    model.fit(B, y_slice)

    # reshape beta parameters
    beta[1] = model.coef_[:, 0].reshape(-1, 1)
    beta[2] = model.coef_[:, 1].reshape(-1, 1)
    beta[3] = model.coef_[:, 2].reshape(-1, 1)

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
    tau_hat = np.stack([beta1_forecast, beta2_forecast, beta3_forecast]) @ B[j-1, :].reshape(-1, 1)

    return tau_hat.item()


def getEHtest(j):
    """
    Calculates the beta(tau) of EH test for a specific maturity j.

    Parameters
    ----------
    j : int
        The index of the maturity to test. (ranges from 2 to 120)

    Returns
    -------
    float
        The beta(tau) for the specified maturity.
    """
    data = pd.read_excel('LW_monthly_1972-2024.xlsx')           # Load the data
    data = data.iloc[:, 2:]                                     # Remove year and month columns
    data.index = range(1, len(data)+1)                          # Reset index to start from 1

    # construct y vector and X matrix
    y = data[j - 1].shift(-1) - data[j]                             # shape (635,)
    X = (data[j] - data[1]) / ((j - 1)/12)                               # shape (635,)

    # fit the regression model
    valid_idx = ~np.isnan(y) & ~np.isnan(X)
    y_clean = y[valid_idx].values.reshape(-1, 1)
    X_clean = X[valid_idx].values.reshape(-1, 1)
    model = LinearRegression(fit_intercept=True)
    model.fit(X_clean, y_clean)

    return model.coef_.item()


def getVasicekPrice(kappa, mu, sigma, r_t, tau):
    """
    Calculates the one factor Vasicek bond price for given parameters.

    Parameters
    ----------
    kappa : float
        The speed of mean reversion.
    mu : float
        The long-term mean level.
    sigma : float
        The volatility of interest rates.
    r_t : float
        The current short-term interest rate.
    tau : int
        The maturity in years.

    Returns
    -------
    float
        The one factor Vasicek bond price for the given parameters.
    """

    B_tau = (np.exp(-kappa * tau) - 1) / kappa
    A_tau = (B_tau + tau) * (sigma**2 / (2 * kappa**2) - mu + 0) - (sigma**2 * B_tau**2) / (4 * kappa)

    P_t_tau = np.exp(A_tau + B_tau * r_t)

    return P_t_tau


def getSimBondOptionPrice(kappa, mu, sigma, r_0, R, delta, T_0, T_1, K):
    """
    Simulates the price of a European bond option using the Vasicek model.

    Parameters
    ----------
    kappa : float
        The speed of mean reversion.
    mu : float
        The long-term mean level.
    sigma : float
        The volatility of interest rates.
    r_0 : float
        The initial short-term interest rate.
    R : int
        The number of simulation paths.
    delta : float
        The time step for the simulation.
    T_0 : float
        The option maturity in years.
    T_1 : float
        The bond maturity in years.
    K : float
        The strike price of the option.

    Returns
    -------
    float
        The simulated price of the European bond option.
    """
    n_steps = T_0 / delta
    std_srt = sigma * np.sqrt((1 - np.exp(-2 * kappa * delta)) / (2 * kappa))

    rates = np.zeros((R, int(n_steps)))
    payoff = []
    # Simulate R paths
    for i in range(R):
        rates[i, 0] = r_0
        # Simulate the short rate path
        for j in range(1, int(n_steps)):
            z = np.random.normal(0, 1)
            rates[i, j] = rates[i, j-1] * np.exp(-kappa * delta) + mu * (1 - np.exp(-kappa * delta)) + std_srt * z

        # Calculate the option payoff at T_0
        discount_factor = np.exp(-np.sum(rates[i, :]) * delta)
        P_T1 = getVasicekPrice(kappa, mu, sigma, rates[i, -1], T_1 - T_0)
        payoff.append(discount_factor * max(P_T1 - K, 0))

    option_price = np.mean(payoff)
    return option_price
