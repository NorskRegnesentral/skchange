"""Data generators."""

from numbers import Number
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sktime.annotation.datagen import piecewise_normal_multivariate


def teeth(
    n_segments: int,
    segment_length: int,
    p: int = 1,
    mean: float = 0.0,
    variance: float = 1.0,
    covariances: np.ndarray = None,
    affected_proportion: float = 1.0,
    random_state: int = None,
) -> pd.DataFrame:
    """
    Generate a DataFrame with teeth-shaped segments.

    Parameters
    ----------
        n_segments : int
            Number of segments to generate.
        segment_length : int
            Length of each segment.
        p : int, optional (default=1)
            Number of dimensions.
        mean : float, optional (default=0.0)
            Mean of each alternating segment.
        variance : float, optional (default=1.0)
            Variances of each alternating segment.
        covariances : array-like, optional (default=None)
            Covariances between dimensions.
        affected_proportion : float, optional (default=1.0)
            Proportion of components {1, ..., p} that are affected by each change.
        random_state : int or RandomState, optional
            Seed or random state for reproducible results. Defaults to None.

    Returns
    -------
        pd.DataFrame: DataFrame with teeth-shaped segments.
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

    segment_lengths = [segment_length] * n_segments
    x = piecewise_normal_multivariate(
        means, segment_lengths, vars, covariances, random_state
    )
    df = pd.DataFrame(x, index=range(len(x)))
    return df


def generate_anomalous_data(
    n: int = 100,
    anomalies: Union[Tuple[int, int], List[Tuple[int, int]]] = (71, 80),
    means: Union[float, List[float], List[np.ndarray]] = 3.0,
    variances: Union[float, List[float], List[np.ndarray]] = 1.0,
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
        random_state : int or RandomState, optional
            Seed or random state for reproducible results. Defaults to None.

    Returns
    -------
        pd.DataFrame: DataFrame with teeth-shaped segments.
    """
    if isinstance(anomalies, Tuple):
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
        df : pd.DataFrame
            DataFrame to add outliers to.
        n_outliers : int
            Number of outliers to add.
        outlier_size : float
            Size of the outliers.

    Returns
    -------
        pd.DataFrame: DataFrame with outliers added.
    """
    outlier_positions = np.linspace(0, df.size - 1, n_outliers, dtype=int)
    df.iloc[outlier_positions] += outlier_size
    return df
