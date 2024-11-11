"""Test statistic for differences in the mean."""

__author__ = ["Tveten"]

import numpy as np
from numba import njit

from skchange.scores.utils import init_sample_sizes, init_sums, init_sums2


@njit(cache=True)
def init_mean_cost(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute sums and weights for `mean_cost`.

    Parameters
    ----------
    X : `np.ndarray`
        2D array.

    Returns
    -------
    sums : `np.ndarray`
        Cumulative sums of `X`.
    sums2 : `np.ndarray`
        Cumulative sums of `X**2`.
    weights : `np.ndarray`
        Weights for `sums2` in the cost calculation.
    """
    return init_sums(X), init_sums2(X), init_sample_sizes(X)


@njit(cache=True)
def mean_cost(
    precomputed_params: tuple[np.ndarray, np.ndarray, np.ndarray],
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Calculate the Gaussian mean likelihood cost for each segment.

    Parameters
    ----------
    precomputed_params : `tuple[np.ndarray, np.ndarray, np.ndarray]`
        Precomputed parameters from `init_mean_cost`.
    starts : `np.ndarray`
        Start indices of the segments.
    ends : `np.ndarray`
        End indices of the segments.

    Returns
    -------
    costs : `np.ndarray`
        Costs for each segment.
    """
    sums, sums2, weights = precomputed_params
    partial_sums = sums[ends + 1] - sums[starts]
    partial_sums2 = sums2[ends + 1] - sums2[starts]
    weights = weights[ends - starts + 1]
    costs = np.sum(partial_sums2 - partial_sums**2 / weights, axis=1)
    return costs


@njit(cache=True)
def init_mean_saving(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute sums and weights for `mean_saving`.

    Parameters
    ----------
    X : `np.ndarray`
        2D array.

    Returns
    -------
    sums : `np.ndarray`
        Cumulative sums of `X`.
    weights : `np.ndarray`
        Weights for `sums2` in the cost calculation.
    """
    return init_sums(X), init_sample_sizes(X)


@njit(cache=True)
def mean_saving(
    precomputed_params: tuple[np.ndarray, np.ndarray],
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """
    Calculate the Gaussian mean likelihood saving for each segment.

    The mean_saving calculates the Gaussian likelihood ratio test statistic of the
    segment starting at `start` and ending at `end` having the maximum likelihod
    estimate of the mean versus zero mean.

    Parameters
    ----------
    precomputed_params : `tuple[np.ndarray, np.ndarray]`
        Precomputed parameters from `init_mean_saving`.
    starts : `np.ndarray`
        Start indices of the segments.
    ends : `np.ndarray`
        End indices of the segments.

    Returns
    -------
    savings : `np.ndarray`
        2D array of savings for each segment (rows) and component (columns).
    """
    sums, weights = precomputed_params
    saving = (sums[ends + 1] - sums[starts]) ** 2 / weights[ends - starts + 1]
    return saving


@njit(cache=True)
def init_mean_score(X: np.ndarray) -> np.ndarray:
    """
    Precompute sums for `mean_score`.

    Parameters
    ----------
    X : `np.ndarray`
        2D array.

    Returns
    -------
    tuple of `np.ndarray`
        Cumulative sums of `X`.
    """
    return init_sums(X)


@njit(cache=True)
def mean_score(
    precomputed_params: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    splits: np.ndarray,
) -> np.ndarray:
    """
    Calculate the CUSUM score for a change in the mean.

    Compares the mean of the data before and after the split within the interval from
    `start:end` (both inclusive).

    Parameters
    ----------
    precomputed_params : `np.ndarray`
        Precomputed parameters from `init_mean_score`.
    starts : `np.ndarray`
        Start indices of the intervals to test for a change in the mean.
    ends : `np.ndarray`
        End indices of the intervals to test for a change in the mean.
    splits : `np.ndarray`
        Split indices of the intervals to test for a change in the mean.

    Returns
    -------
    `np.ndarray`
        Scores for a difference in the mean at the given intervals and splits.

    Notes
    -----
    To optimize performance, no checks are performed on (`starts`, `splits`, `ends`).
    """
    sums = precomputed_params
    before_sum = sums[splits + 1] - sums[starts]
    before_weight = np.sqrt(
        (ends - splits) / ((ends - starts + 1) * (splits - starts + 1))
    )
    before_weight = before_weight.reshape(-1, 1)
    after_sum = sums[ends + 1] - sums[splits + 1]
    after_weight = np.sqrt(
        (splits - starts + 1) / ((ends - starts + 1) * (ends - splits))
    )
    after_weight = after_weight.reshape(-1, 1)
    return np.sum(np.abs(after_weight * after_sum - before_weight * before_sum), axis=1)


@njit(cache=True)
def init_mean_anomaly_score(X: np.ndarray) -> np.ndarray:
    """
    Precompute sums for `mean_score`.

    Parameters
    ----------
    X : `np.ndarray`
        2D array.

    Returns
    -------
    `np.ndarray`
        Cumulative sums of `X`.
    """
    return init_sums(X)


@njit(cache=True)
def mean_anomaly_score(
    precomputed_params: np.ndarray,
    interval_starts: np.ndarray,
    interval_ends: np.ndarray,
    anomaly_starts: np.ndarray,
    anomaly_ends: np.ndarray,
) -> np.ndarray:
    """
    Calculate the CUSUM score for difference in the mean of a subinterval.

    Compares the mean of the data in `anomaly_start:anomaly_end` (both inclusive)
    to the mean of the complement of `interval_start:interval_end` (both inclusive) and
    `anomaly_start:anomaly_end`.

    The overall and anomalous intervals must satisfy
    `interval_start > anomaly_start <= anomaly_end <= interval_end`.

    Parameters
    ----------
    precomputed_params : `np.ndarray`
        Precomputed parameters from `init_mean_score`.
    interval_starts : `np.ndarray`
        Start indices of the intervals to test for an anomaly in.
    interval_ends : `np.ndarray`
        End indices of the intervals to test for an anomaly in.
    anomaly_starts : `np.ndarray`
        Start indices of the anomalies.
    anomaly_ends : `np.ndarray`
        End indices of the anomalies.

    Returns
    -------
    `np.ndarray`
        Score for a difference in the mean.

    Notes
    -----
    To optimize performance, no checks are performed on the inputs.
    """
    sums = precomputed_params
    baseline_sum = (
        sums[interval_ends + 1]
        - sums[anomaly_ends + 1]
        + sums[anomaly_starts]
        - sums[interval_starts]
    )
    baseline_n = interval_ends - anomaly_ends + anomaly_starts - interval_starts
    baseline_n = baseline_n.reshape(-1, 1)
    baseline_mean = baseline_sum / baseline_n
    anomaly_sum = sums[anomaly_ends + 1] - sums[anomaly_starts]
    anomaly_n = (anomaly_ends - anomaly_starts + 1).reshape(-1, 1)
    anomaly_mean = anomaly_sum / anomaly_n
    weight = (1 / anomaly_n + 1 / baseline_n) ** (-1 / 2)
    return np.sum(np.abs(weight * (baseline_mean - anomaly_mean)), axis=1)
