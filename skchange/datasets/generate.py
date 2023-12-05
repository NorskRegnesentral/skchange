import numpy as np
import pandas as pd
from sktime.annotation.datagen import piecewise_normal_multivariate


def teeth(
    n_segments,
    mean_size,
    segment_length,
    p=1,
    variances=1.0,
    covariances=None,
    random_state=None,
) -> pd.DataFrame:
    means = []
    for i in range(n_segments):
        mean = [0] * p if i % 2 == 0 else [mean_size] * p
        means.append(mean)
    segment_lengths = [segment_length] * n_segments
    x = piecewise_normal_multivariate(
        means, segment_lengths, variances, covariances, random_state
    )
    df = pd.DataFrame(x, index=range(len(x)))
    return df


def add_linspace_outliers(df, n_outliers, outlier_size):
    outlier_positions = np.linspace(0, df.size - 1, n_outliers, dtype=int)
    df.iloc[outlier_positions] += outlier_size
    return df
