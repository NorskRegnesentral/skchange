"""Data generators."""

import numpy as np
import pandas as pd
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
