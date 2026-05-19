"""Data generators for regression data."""

import numpy as np
from sklearn.datasets import make_regression

from ._utils import check_random_generator, check_segment_lengths


def generate_piecewise_regression_data(
    lengths: int | list[int] | np.ndarray | None = None,
    *,
    n_segments: int = 3,
    n_samples: int = 100,
    n_features: int = 1,
    n_informative: int = 1,
    n_targets: int = 1,
    bias: float = 0.0,
    effective_rank: int | None = None,
    tail_strength: float = 0.5,
    noise: float = 1.0,
    shuffle: bool = True,
    seed: int | np.random.Generator | None = None,
    return_params: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]:
    """Generate piecewise linear regression data.

    Generate independent segments of data from `sklearn.datasets.make_regression`.

    Parameters
    ----------
    lengths : int, list of int or np.ndarray, optional (default=None)
        The segment lengths. There are three possible cases:

        1. `list` or `numpy array`: Custom set of segment lengths.
        2. `int`: Length of `n_segments` equal segments.
        3. `None`: Generate `n_segments` random segment lengths with a total sample size
           of `n_samples`.

    n_segments : int (default=3)
        Number of segments to generate if `lengths` is an integer or None.

    n_samples : int (default=100)
        Total number of samples to generate if `lengths` is not specified.

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

    seed : np.random.Generator | int | None, optional
        Seed for the random number generator or a numpy random generator instance.
        If specified, this ensures reproducible output across multiple calls.

    return_params : bool
        Whether to return the parameters used for data generation.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        The generated feature matrix.
    y : np.ndarray of shape (n_samples, n_targets)
        The generated target matrix.
    dict, optional
        If `return_params` is True, a dictionary containing the parameters used to
        generate the data, including segment lengths, coefficients, change points,
        total number of samples and total number of segments.

    """
    random_generator = check_random_generator(seed)
    lengths = check_segment_lengths(
        lengths, n_segments, n_samples, seed=random_generator
    )
    n_segments = len(lengths)
    n_samples = int(np.sum(lengths))

    # make_regression requires a np.random.RandomState instance.
    random_generator = np.random.RandomState(random_generator.integers(0, 2**32 - 1))

    ends = np.cumsum(lengths)
    starts = np.concatenate(([0], ends[:-1]))
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
            random_state=random_generator,
        )
        generated_x[start:end, :] = x
        generated_y[start:end, :] = y.reshape(segment_length, n_targets)
        coefs.append(coef)

    if return_params:
        params = {
            "n_segments": n_segments,
            "n_samples": n_samples,
            "lengths": lengths,
            "coefs": coefs,
            "change_points": starts[1:],
        }
        return generated_x, generated_y, params

    return generated_x, generated_y
