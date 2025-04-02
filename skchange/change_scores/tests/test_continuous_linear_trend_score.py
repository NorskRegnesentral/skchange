import numpy as np
import pytest

from skchange.change_scores import ContinuousLinearTrendScore


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


def linear_trend_score(
    starts: np.ndarray, splits: np.ndarray, ends: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Evaluate the continuous linear trend cost.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the first intervals (inclusive).
    splits : np.ndarray
        Split indices between the intervals (contained in second interval).
    ends : np.ndarray
        End indices of the second intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    ### NOTE: Assume 'time' is index of the data. i.e. time = 0, 1, 2, ..., len(X)-1
    ###       This assumption could be changed later.
    n_intervals = len(starts)
    n_columns = X.shape[1]
    scores = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, split, end = starts[i], splits[i], ends[i]
        split_interval_trend_data = np.zeros((end - start, 3))
        split_interval_trend_data[:, 0] = 1.0  # Intercept
        # Whole interval slope:
        split_interval_trend_data[:, 1] = np.arange(end - start)  # Time steps

        # Change in slope 'split + 1' index:
        # Continuous at the first point of the second interal, [split, end - 1]:
        # trend data index starts at 0 from 'start'.
        # split_interval_trend_data[(split - start) :, 2] = np.arange(end - split)

        # THIS IS WHAT the 'NOT' people DO:
        # Change in slope from 'split' index:
        # Continuous in the last point of the first interval, [start, split - 1]:
        # trend data index starts at 0 from 'start'.
        split_interval_trend_data[(split - start) :, 2] = np.arange(1, end - split + 1)

        # Calculate the slope and intercept for the whole interval:
        split_interval_linreg_res = np.linalg.lstsq(
            split_interval_trend_data, X[start:end, :]
        )
        split_interval_squared_residuals = split_interval_linreg_res[1]

        # By only regressing onto the first two columns, we can calculate the cost
        # without allowing for a change in slope at the split point.
        joint_interval_linreg_res = np.linalg.lstsq(
            split_interval_trend_data[:, [0, 1]], X[start:end, :]
        )
        joint_interval_squared_residuals = joint_interval_linreg_res[1]

        scores[i, :] = (
            joint_interval_squared_residuals - split_interval_squared_residuals
        )

    return scores


def regression_based_piecewise_linear_trend_score(
    signal: np.ndarray, start: int, split: int, end: int
):
    pass
