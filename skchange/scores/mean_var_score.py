"""Test statistic for differences in the mean and/or variance."""

__author__ = ["Tveten"]

import numpy as np
from numba import njit

from skchange.utils.numba.general import truncate_below
from skchange.utils.numba.stats import col_cumsum


@njit(cache=True)
def init_mean_var_score(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precompute sums and squared sums for `mean_var_score`.

    Parameters
    ----------
    X : np.ndarray
        2D array.

    Returns
    -------
    sums : np.ndarray
        Cumulative sums of `X`.
    """
    n = X.shape[0]
    p = X.shape[1]
    # 0.0 as first row to make calculations work also for start = 0 in mean_score
    sums = np.zeros((n + 1, p))
    sums[1:] = col_cumsum(X)
    sums2 = np.zeros((n + 1, p))
    sums2[1:] = col_cumsum(X**2)
    return sums, sums2


@njit(cache=True)
def var_from_sums(sums1: np.ndarray, sums2: np.ndarray, i: np.ndarray, j: np.ndarray):
    """Calculate variance from precomputed sums.

    Parameters
    ----------
    sums1 : np.ndarray
        Cumulative sums of `X`.
    sums2 : np.ndarray
        Cumulative sums of `X**2`.
    i : np.ndarray
        Start indices in the original data `X`.
    j : np.ndarray
        End indices in the original data `X`.

    Returns
    -------
    var : float
        Variance of `X[i:j]` (inclusive `j`).
    """
    n = (j - i + 1).reshape(-1, 1)
    sum1 = sums1[j + 1] - sums1[i]  # Indices is moved one step forward
    sum2 = sums2[j + 1] - sums2[i]  # Indices is moved one step forward
    var = sum2 / n - (sum1 / n) ** 2
    return truncate_below(var, 1e-16)  # standard deviation lower bound of 1e-8


@njit(cache=True)
def mean_var_score(
    precomputed_params: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    splits: np.ndarray,
) -> np.ndarray:
    """Calculate the score for a change in the mean and/or variance.

    Computes the likelihood ratio test for a change in the mean and/or variance of
    i.i.d. Gaussian data.

    Parameters
    ----------
    precomputed_params : np.ndarray
        Precomputed parameters from `init_mean_var_score`.
    starts : np.ndarray
        Start indices of the intervals to test for a change in the mean.
    ends : np.ndarray
        End indices of the intervals to test for a change in the mean.
    splits : np.ndarray
        Split indices of the intervals to test for a change in the mean.

    Returns
    -------
    score : np.ndarray
        Scores for a difference in the mean or variance at the given intervals and
        splits.

    Notes
    -----
    To optimize performance, no checks are performed on (`starts`, `splits`, `ends`).
    """
    sums, sums2 = precomputed_params
    before_var = var_from_sums(sums, sums2, starts, splits)
    after_var = var_from_sums(sums, sums2, splits + 1, ends)
    full_var = var_from_sums(sums, sums2, starts, ends)
    before_n = (splits - starts + 1).reshape(-1, 1)
    after_n = (ends - splits).reshape(-1, 1)
    before_term = -before_n * np.log(before_var / full_var)
    after_term = -after_n * np.log(after_var / full_var)
    likelihood_ratio = before_term + after_term
    return np.sum(likelihood_ratio, axis=1)


@njit(cache=True)
def baseline_var_from_sums(
    sums1: np.ndarray,
    sums2: np.ndarray,
    interval_start: np.ndarray,
    interval_end: np.ndarray,
    anomaly_start: np.ndarray,
    anomaly_end: np.ndarray,
):
    """Calculate variance from precomputed sums.

    Parameters
    ----------
    sums1 : np.ndarray
        Cumulative sums of `X`.
    sums2 : np.ndarray
        Cumulative sums of `X**2`.
    interval_start : np.ndarray
        Start indices in the original data `X`.
    interval_end : np.ndarray
        End indices in the original data `X`.
    anomaly_start : np.ndarray
        Start indices of the anomalies.
    anomaly_end : np.ndarray
        End indices of the anomalies.

    Returns
    -------
    var : float
        Variance of `X` in the specified intervals.
    """
    n = (interval_end - anomaly_end + anomaly_start - interval_start).reshape(-1, 1)
    sum1 = (
        sums1[interval_end + 1]
        - sums1[anomaly_end + 1]
        + sums1[anomaly_start]
        - sums1[interval_start]
    )
    sum2 = (
        sums2[interval_end + 1]
        - sums2[anomaly_end + 1]
        + sums2[anomaly_start]
        - sums2[interval_start]
    )
    var = sum2 / n - (sum1 / n) ** 2
    return truncate_below(var, 1e-16)  # standard deviation lower bound of 1e-8


@njit(cache=True)
def mean_var_anomaly_score(
    precomputed_params: np.ndarray,
    interval_starts: np.ndarray,
    interval_ends: np.ndarray,
    anomaly_starts: np.ndarray,
    anomaly_ends: np.ndarray,
) -> np.ndarray:
    """Calculate the score for an anomaly in the mean and/or variance.

    Computes the likelihood ratio test for a change in the mean and/or variance of
    i.i.d. Gaussian data between the anomaly interval and its complement within
    the overall interval.

    The overall and anomalous intervals must satisfy
    `interval_start > anomaly_start <= anomaly_end <= interval_end`.

    Parameters
    ----------
    precomputed_params : np.ndarray
        Precomputed parameters from `init_mean_var_score`.
    interval_starts : np.ndarray
        Start indices of the intervals to test for an anomaly in.
    interval_ends : np.ndarray
        End indices of the intervals to test for an anomaly in.
    anomaly_starts : np.ndarray
        Start indices of the anomalies.
    anomaly_ends : np.ndarray
        End indices of the anomalies.

    Returns
    -------
    score : float
        Score for an anomaly in the mean and/or variance.

    Notes
    -----
    To optimize performance, no checks are performed on the inputs.
    """
    sums, sums2 = precomputed_params

    baseline_ns = interval_ends - anomaly_ends + anomaly_starts - interval_starts
    baseline_ns = baseline_ns.reshape(-1, 1)
    anomaly_ns = (anomaly_ends - anomaly_starts + 1).reshape(-1, 1)

    baseline_vars = baseline_var_from_sums(
        sums, sums2, interval_starts, interval_ends, anomaly_starts, anomaly_ends
    )
    anomaly_vars = var_from_sums(sums, sums2, anomaly_starts, anomaly_ends)
    full_vars = var_from_sums(sums, sums2, interval_starts, interval_ends)

    baseline_term = -baseline_ns * np.log(baseline_vars / full_vars)
    anomaly_term = -anomaly_ns * np.log(anomaly_vars / full_vars)

    likelihood_ratio = baseline_term + anomaly_term
    return np.sum(likelihood_ratio, axis=1)
