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
        p : int, optional
            Number of dimensions. Defaults to 1.
        mean : float, optional
            Mean of each alternating segment.
        variance : float, optional
            Variances of each alternating segment. Defaults to 1.0.
        covariances : array-like, optional
            Covariances between dimensions. Defaults to None.
        random_state : int or RandomState, optional
            Seed or random state for reproducible results. Defaults to None.

    Returns
    -------
        pd.DataFrame: DataFrame with teeth-shaped segments.
    """
    means = []
    vars = []
    for i in range(n_segments):
        mean_vec = [0] * p if i % 2 == 0 else [mean] * p
        means.append(mean_vec)
        vars_vec = [1] * p if i % 2 == 0 else [variance] * p
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
