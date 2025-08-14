"""Data generators for regression data."""

import numbers

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from ._generate import check_random_state


def generate_piecewise_regression_data(
    lengths: int | list[int] | np.ndarray = 100,
    *,
    n_features: int = 1,
    n_informative: int = 1,
    n_targets: int = 1,
    bias: float = 0.0,
    effective_rank: int | None = None,
    tail_strength: float = 0.5,
    noise: float = 1.0,
    shuffle: bool = True,
    random_state: int | np.random.Generator | None = None,
    return_params: bool = False,
) -> (
    tuple[pd.DataFrame, list[str], list[str]]
    | tuple[pd.DataFrame, list[str], list[str], dict]
):
    """Generate piecewise linear regression data.

    Generate independent segments of data from `sklearn.datasets.make_regression`.

    Parameters
    ----------
    lengths : int | list[int] | np.ndarray
        The number of samples in each segment. If a single integer is provided,
        a single segment of that length is generated.

    n_features : int
        The total number of features.

    n_informative : int
        The number of informative features.

    n_targets : int
        The number of target variables.

    bias : float
        The bias term in the linear model. Used across all segments.

    effective_rank : int | None
        The effective rank of the feature matrix. Used across all segments.

    tail_strength : float
        The tail strength of the noise distribution. Used across all segments.

    noise : float
        The standard deviation of the Gaussian noise applied to the output.
        Used across all segments.

    shuffle : bool
        Whether to shuffle the samples and features per segment.

    random_state : int | np.random.Generator | None
        The random seed or generator for reproducibility.

    return_params : bool
        Whether to return the parameters used for data generation.

    Returns
    -------
    tuple[pd.DataFrame, list[str], list[str]]
        A DataFrame containing the generated data, a list of feature column names,
        and a list of target column names.

    tuple[pd.DataFrame, list[str], list[str], dict]
        If `return_params` is True, also returns a dictionary with the parameters used
        for generating the data, including segment lengths, coefficients, change points,
        and total number of samples.

    """
    random_state = check_random_state(random_state)
    # make_regression requires a np.random.RandomState instance.
    random_state = np.random.RandomState(random_state.integers(0, 2**32 - 1))

    if isinstance(lengths, numbers.Integral):
        lengths = [lengths]
    if any([length <= 0 for length in lengths]):
        raise ValueError("All segment lengths must be positive integers.")

    n_segments = len(lengths)
    if n_segments < 1:
        raise ValueError("At least one segment length must be provided.")

    ends = np.cumsum(lengths)
    starts = np.concatenate(([0], ends[:-1]))
    n_samples = ends[-1]
    generated_x = np.empty((n_samples, n_features), dtype=np.float64)
    generated_y = np.empty((n_samples, n_targets), dtype=np.float64)
    coefs = []
    for start, end in zip(starts, ends):
        segment_length = end - start
        x, y, coef = make_regression(
            n_samples=segment_length,
            n_features=n_features,
            n_informative=n_informative,
            n_targets=n_targets,
            bias=bias,
            effective_rank=effective_rank,
            tail_strength=tail_strength,
            noise=noise,
            shuffle=shuffle,
            coef=True,
            random_state=random_state,
        )
        generated_x[start:end, :] = x
        generated_y[start:end, :] = y.reshape(segment_length, n_targets)
        coefs.append(coef)

    feature_cols = [f"feature_{i}" for i in range(n_features)]
    target_cols = [f"target_{i}" for i in range(n_targets)]
    generated_df = pd.DataFrame(
        np.concatenate((generated_x, generated_y), axis=1),
        columns=feature_cols + target_cols,
    )

    if return_params:
        params = {
            "lengths": lengths,
            "coefs": coefs,
            "change_points": starts[1:],
            "n_samples": n_samples,
        }
        return generated_df, feature_cols, target_cols, params

    return generated_df, feature_cols, target_cols
