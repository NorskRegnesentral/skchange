"""Data generators."""

import numpy as np
import pandas as pd
from sktime.annotation.datagen import piecewise_normal_multivariate


def teeth(
    n_segments,
    mean,
    segment_length,
    p=1,
    variances=1.0,
    covariances=None,
    random_state=None,
) -> pd.DataFrame:
    """
    Generate a DataFrame with teeth-shaped segments.

    Parameters
    ----------
        n_segments : int
            Number of segments to generate.
        mean : float
            Mean of each alternating segment.
        segment_length : float
            Length of each segment.
        p : int, optional
            Number of dimensions. Defaults to 1.
        variances : float or array-like, optional
            Variances of the segments. Defaults to 1.0.
        covariances : array-like, optional
            Covariances between dimensions. Defaults to None.
        random_state : int or RandomState, optional
            Seed or random state for reproducible results. Defaults to None.

    Returns
    -------
        pd.DataFrame: DataFrame with teeth-shaped segments.
    """
    means = []
    for i in range(n_segments):
        mean = [0] * p if i % 2 == 0 else [mean] * p
        means.append(mean)
    segment_lengths = [segment_length] * n_segments
    x = piecewise_normal_multivariate(
        means, segment_lengths, variances, covariances, random_state
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
