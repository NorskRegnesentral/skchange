"""Data generators."""

__author__ = ["Tveten"]

from numbers import Number
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def generate_changing_data(
    n: int = 100,
    changepoints: Union[int, list[int]] = 49,
    means: Union[float, list[float], list[np.ndarray]] = 3.0,
    variances: Union[float, list[float], list[np.ndarray]] = 1.0,
    random_state: int = None,
):
    """
    Generate piecewise multivariate normal data with changing means and variances.

    Parameters
    ----------
    n : int, optional, default=100
        Number of observations.
    changepoints : int or list of ints, optional, default=50
        Changepoints in the data.
    means : list of floats or list of arrays, optional, default=0.0
        List of means for each segment.
    variances : list of floats or list of arrays, optional, default=1.0
        List of variances for each segment.
    random_state : int or `RandomState`, optional
        Seed or random state for reproducible results. Defaults to None.

    Returns
    -------
    `pd.DataFrame`
        DataFrame with generated data.
    """
    if isinstance(changepoints, int):
        changepoints = [changepoints]
    if isinstance(means, Number):
        means = [means]
    if isinstance(variances, Number):
        variances = [variances]

    if isinstance(means[0], Number):
        means = [np.array([mean]) for mean in means]
    if isinstance(variances[0], Number):
        variances = [np.array([variance]) for variance in variances]

    if len(means) == 1:
        means = means * len(changepoints + 1)
    if len(variances) == 1:
        variances = variances * len(changepoints + 1)

    if len(changepoints) != len(means) - 1 or len(changepoints) != len(variances) - 1:
        raise ValueError(
            "Number of segments (len(changepoints) + 1),"
            + " means and variances must be the same."
        )
    if any([changepoint > n - 1 for changepoint in changepoints]):
        raise ValueError(
            "Changepoints must be within the range of the data"
            + f" (n={n} and max(changepoints)={max(changepoints)})."
        )

    p = len(means[0])
    x = multivariate_normal.rvs(np.zeros(p), np.eye(p), n, random_state)
    changepoints = [-1] + changepoints + [n - 1]
    for prev_cpt, next_cpt, mean, variance in zip(
        changepoints[:-1], changepoints[1:], means, variances
    ):
        x[prev_cpt + 1 : next_cpt + 1] = (
            mean + np.sqrt(variance) * x[prev_cpt + 1 : next_cpt + 1]
        )

    df = pd.DataFrame(x, index=range(len(x)))
    return df


def generate_anomalous_data(
    n: int = 100,
    anomalies: Union[tuple[int, int], list[tuple[int, int]]] = (71, 80),
    means: Union[float, list[float], list[np.ndarray]] = 3.0,
    variances: Union[float, list[float], list[np.ndarray]] = 1.0,
    random_state: int = None,
) -> pd.DataFrame:
    """
    Generate multivariate normal data with anomalies.

    Parameters
    ----------
    n : int, optional (default=100)
        Number of observations.
    anomalies : list of tuples, optional (default=[(71, 80)])
        List of tuples of the form (start, end) indicating the start and end of an
        anomaly.
    means : list of floats or list of arrays, optional (default=[0.0])
        List of means for each segment.
    variances : list of floats or list of arrays, optional (default=[1.0])
        List of variances for each segment.
    random_state : int or `RandomState`, optional
        Seed or random state for reproducible results. Defaults to None.

    Returns
    -------
    `pd.DataFrame`
        DataFrame with generated data.
    """
    if isinstance(anomalies, tuple):
        anomalies = [anomalies]
    if isinstance(means, Number):
        means = [means]
    if isinstance(variances, Number):
        variances = [variances]

    if isinstance(means[0], Number):
        means = [np.array([mean]) for mean in means]
    if isinstance(variances[0], Number):
        variances = [np.array([variance]) for variance in variances]

    if len(means) == 1:
        means = means * len(anomalies)
    if len(variances) == 1:
        variances = variances * len(anomalies)

    if len(anomalies) != len(means) or len(anomalies) != len(variances):
        raise ValueError("Number of anomalies, means and variances must be the same.")
    if any([len(anomaly) != 2 for anomaly in anomalies]):
        raise ValueError("Anomalies must be of length 2.")
    if any([anomaly[0] > anomaly[1] for anomaly in anomalies]):
        raise ValueError("The start of an anomaly must be before its end.")
    if any([anomaly[1] > n - 1 for anomaly in anomalies]):
        raise ValueError("Anomalies must be within the range of the data.")

    p = len(means[0])
    x = multivariate_normal.rvs(np.zeros(p), np.eye(p), n, random_state)
    for anomaly, mean, variance in zip(anomalies, means, variances):
        start, end = anomaly
        x[start : end + 1] = mean + np.sqrt(variance) * x[start : end + 1]
    df = pd.DataFrame(x, index=range(len(x)))
    return df


def add_linspace_outliers(df, n_outliers, outlier_size):
    """
    Add outliers to a DataFrame at evenly spaced positions.

    Parameters
    ----------
    df : `pd.DataFrame`
        DataFrame to add outliers to.
    n_outliers : int
        Number of outliers to add.
    outlier_size : float
        Size of the outliers.

    Returns
    -------
    `pd.DataFrame`
        DataFrame with outliers added.
    """
    outlier_positions = np.linspace(0, df.size - 1, n_outliers, dtype=int)
    df.iloc[outlier_positions] += outlier_size
    return df


def generate_alternating_data(
    n_segments: int,
    segment_length: int,
    p: int = 1,
    mean: float = 0.0,
    variance: float = 1.0,
    affected_proportion: float = 1.0,
    random_state: int = None,
) -> pd.DataFrame:
    """
    Generate multivariate normal data that is alternating between two states.

    The data alternates between a state with mean 0 and variance 1 and a state with
    mean `mean` and variance `variance`. The length of the segments are all identical
    and equal to `segment_length`. The proportion of components that are affected by
    the change is determined by `affected_proportion`.

    Parameters
    ----------
    n_segments : int
        Number of segments to generate.
    segment_length : int
        Length of each segment.
    p : int, optional (default=1)
        Number of dimensions.
    mean : float, optional (default=0.0)
        Mean of every other segment.
    variance : float, optional (default=1.0)
        Variances of every other segment.
    affected_proportion : float, optional (default=1.0)
        Proportion of components {1, ..., p} that are affected by each change in
        every other segment.
    random_state : int or `RandomState`, optional
        Seed or random state for reproducible results. Defaults to None.

    Returns
    -------
    `pd.DataFrame`
        DataFrame with generated data.
    """
    means = []
    vars = []
    n_affected = int(np.round(p * affected_proportion))
    for i in range(n_segments):
        zero_mean = [0] * p
        changed_mean = [mean] * n_affected + [0] * (p - n_affected)
        mean_vec = zero_mean if i % 2 == 0 else changed_mean
        means.append(mean_vec)
        one_var = [1] * p
        changed_var = [variance] * n_affected + [1] * (p - n_affected)
        vars_vec = one_var if i % 2 == 0 else changed_var
        vars.append(vars_vec)

    n = segment_length * n_segments
    changepoints = [segment_length * i - 1 for i in range(1, n_segments)]
    x = generate_changing_data(n, changepoints, means, vars, random_state)
    df = pd.DataFrame(x, index=range(len(x)))
    return df
