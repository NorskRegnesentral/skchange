import numpy as np

# Clever idea, use least squares to fit a continuous piecewise linear function.
# Continuity is enforced by using the same intercept at the kink point.


def fit_indexed_linear_trend(xs: np.ndarray) -> tuple[float, float]:
    """Calculate the optimal linear trend for a given array.

    Assuming the time steps are [0, 1, 2, ..., n-1], we can optimize the calculation
    of the least squares intercept and slope.

    Parameters
    ----------
    xs : np.ndarray
        1D array of data points

    Returns
    -------
    tuple
        (slope, intercept) of the best-fit line
    """
    n_samples = len(xs)

    # For evenly spaced time steps [0, 1, 2, ..., n-1],
    # the mean time step is (n-1)/2.
    mean_t = (n_samples - 1) / 2.0

    # Optimized calculation for denominator:
    # sum of (t - mean_t)^2 = n*(n^2-1)/12
    denominator = n_samples * (n_samples * n_samples - 1) / 12.0

    # Calculate numerator: sum((t-mean_t)*(x-mean_x))
    # numerator = np.sum((np.arange(n) - mean_t) * (xs - mean_x))
    mean_x = np.mean(xs)
    numerator = 0.0
    for i in range(n_samples):
        numerator += (i - mean_t) * (xs[i] - mean_x)

    slope = numerator / denominator
    intercept = mean_x - slope * mean_t

    return intercept, slope


# %%
# Clever idea, use least squares to fit a continuous piecewise linear function.
# Continuity is enforced by using the same intercept at the kink point.
def optimal_continuous_piecewise_linear_v2(signal, m):
    """
    Find the optimal continuous piecewise linear trend with a kink at index m.

    This function fits a continuous piecewise linear function with a single kink
    at index m that minimizes the squared error to the signal.

    Parameters
    ----------
    signal : np.ndarray
        1D array containing the signal
    m : int
        Index where the kink should be placed (0 < m < len(signal)-1)

    Returns
    -------
    fitted_trend : np.ndarray
        Fitted continuous piecewise linear trend values
    params : tuple
        (intercept1, slope1, slope2) parameters of the piecewise linear function
    error : float
        Sum of squared errors between the signal and fitted trend
    """
    n = len(signal)
    if m <= 0 or m >= n - 1:
        raise ValueError("m must be between 1 and len(signal)-2")

    # Create design matrix X for the piecewise linear regression
    # We need three parameters: intercept, slope1, and slope2
    X = np.zeros((n, 3))

    # First column is all ones (intercept)
    X[:, 0] = 1

    # Second column is the time index for the first segment
    X[:, 1] = np.arange(n)

    # Third column represents the additional slope for the second segment
    # For continuity at x=m:
    #     f(x) = intercept + slope1*m + slope2*max(0,x-m)
    # For x < m: f(x) = intercept + slope1*x
    # For x > m: f(x) = intercept + slope1*m + slope2*(x-m)
    # At x=m: intercept + slope1*m + slope2*max(0,x-m)
    # At x=m: intercept + slope1*m + slope2*max(0,m-m),
    #          = intercept + slope1*m + slope2*0,
    #          = intercept + slope1*m.
    # At x=m + 1: intercept + slope1*(m + 1) + slope2*max(0,x-m)
    X[m:, 2] = np.arange(n - m)

    # Solve for the parameters using least squares
    beta, residuals, rank, singular_values = np.linalg.lstsq(X, signal)

    # Extract parameters
    intercept = beta[0]
    slope1 = beta[1]
    slope2 = beta[2]  # Additional slope (total slope for segment 2 is slope1 + slope2)

    # Compute the fitted values
    # fitted_trend = np.zeros(n)
    # fitted_trend[: m + 1] = intercept + slope1 * np.arange(m + 1)
    # fitted_trend[m:] = (
    #     intercept + slope1 * m + slope1 * np.arange(n - m) + slope2 * np.arange(n - m)
    # )
    fitted_trend = X @ beta

    # Calculate error
    error = np.sum(np.square(signal - fitted_trend))
    print("Residuals:", residuals)
    print("Error:", error)
    print("Rank:", rank)
    print("Singular values:", singular_values)

    return fitted_trend, (intercept, slope1, slope1 + slope2), error


# %%
# Test the functions:
if __name__ == "__main__":
    # Example signal with a change point
    signal = np.array([1, 2, 3, 4, 5, 7, 9, 11, 13, 15])

    # Find the optimal continuous piecewise linear trend with a kink at index 5
    # Test the second version
    fitted_trend_v2, params_v2, error_v2 = optimal_continuous_piecewise_linear_v2(
        signal, 5
    )
    print("Fitted trend (v2):", fitted_trend_v2)
    print("Parameters (v2):", params_v2)
    print("Error (v2):", error_v2)

    full_segment_params = fit_indexed_linear_trend(signal)
    print("Full segment params:", full_segment_params)
    full_segment_trend_residuals = signal - (
        full_segment_params[0] + np.arange(len(signal)) * full_segment_params[1]
    )
    full_segment_X = np.vstack((np.ones(len(signal)), np.arange(len(signal)))).T
    full_trend_lst_sq_res = np.linalg.lstsq(
        full_segment_X, signal, rcond=None
    )  # This should give the same result as the above calculation.


# %%
