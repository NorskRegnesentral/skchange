"""Test statistic for differences in the mean."""

import numpy as np
from numba import njit

from skchange.utils.numba.stats import col_cumsum


@njit(cache=True)
def init_mean_score(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precompute sums for mean_score.

    Parameters
    ----------
    X : np.ndarray
        2D array.

    Returns
    -------
    sums : np.ndarray
        Cumulative sums of X.
    """
    n = X.shape[0]
    p = X.shape[1]
    # 0.0 as first row to make calculations work also for start = 0 in mean_score
    sums = np.zeros((n + 1, p))
    sums[1:] = col_cumsum(X)
    return sums


@njit(cache=True)
def mean_score(
    precomputed_params: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    splits: np.ndarray,
) -> np.ndarray:
    """Calculate the CUSUM score for a change in the mean.

    Compares the mean of the data before and after the split within the interval from
    start:end (both inclusive).

    Parameters
    ----------
    precomputed_params : np.ndarray
        Precomputed parameters from init_mean_score.
    start : np.ndarray
        Start indices of the intervals to test for a change in the mean.
    end : np.ndarray
        End indices of the intervals to test for a change in the mean.
    split : np.ndarray
        Split indices of the intervals to test for a change in the mean.

    Returns
    -------
    score : np.ndarray
        Scores for a difference in the mean at the given intervals and splits

    Notes
    -----
    To optimize performance, no checks are performed on (start, split, end).
    """
    sums = precomputed_params
    before_sum = sums[splits + 1] - sums[starts]
    before_weight = np.sqrt((ends - splits) / ((ends - starts + 1) * (splits - starts + 1)))
    before_weight = before_weight.reshape(-1, 1)
    after_sum = sums[ends + 1] - sums[splits + 1]
    after_weight = np.sqrt((splits - starts + 1) / ((ends - starts + 1) * (ends - splits)))
    after_weight = after_weight.reshape(-1, 1)
    return np.sum(np.abs(after_weight * after_sum - before_weight * before_sum), axis=1)


@njit(cache=True)
def mean_anomaly_score(
    precomputed_params: np.ndarray,
    interval_start: np.ndarray,
    interval_end: np.ndarray,
    anomaly_start: np.ndarray,
    anomaly_end: np.ndarray,
) -> np.ndarray:
    """Calculate the CUSUM score for difference in the mean of a subinterval.

    Compares the mean of the data in anomaly_start:anomaly_end (both inclusive)
    to the mean of the complement of interval_start:interval_end (both inclusive) and
    anomaly_start:anomaly_end.

    The overall and anomalous intervals must satisfy
    `interval_start > anomaly_start <= anomaly_end <= interval_end`

    Parameters
    ----------
    precomputed_params : np.ndarray
        Precomputed parameters from init_mean_score.
    interval_start : np.ndarray
        Start indices of the intervals to test for an anomaly in.
    interval_end : np.ndarray
        End indices of the intervals to test for an anomaly in.
    anomaly_start : np.ndarray
        Start indices of the anomalies.
    anomaly_end : np.ndarray
        End indices of the anomalies.

    Returns
    -------
    score : float
        Score for a difference in the mean.

    Notes
    -----
    To optimize performance, no checks are performed on the inputs.
    """
    sums = precomputed_params
    baseline_sum = (
        sums[interval_end + 1]
        - sums[anomaly_end + 1]
        + sums[anomaly_start]
        - sums[interval_start]
    )
    baseline_n = interval_end - anomaly_end + anomaly_start - interval_start
    baseline_n = baseline_n.reshape(-1, 1)
    baseline_mean = baseline_sum / baseline_n
    anomaly_sum = sums[anomaly_end + 1] - sums[anomaly_start]
    anomaly_n = (anomaly_end - anomaly_start + 1).reshape(-1, 1)
    anomaly_mean = anomaly_sum / anomaly_n
    weight = (1 / anomaly_n + 1 / baseline_n) ** (-1 / 2)
    return np.sum(np.abs(weight * (baseline_mean - anomaly_mean)), axis=1)
