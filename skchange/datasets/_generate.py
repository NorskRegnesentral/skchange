"""Data generators."""

__author__ = ["Tveten"]

from numbers import Number

import numpy as np
import pandas as pd
import scipy.stats


def check_random_state(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    """Check and return a random state.

    Parameters
    ----------
    random_state : int or np.random.Generator or None
        Seed for the random number generator. If None, a new random state is created.
        If an int, a new Generator is created with that seed.

    Returns
    -------
    np.random.Generator
        Random state to use for random number generation.
    """
    if random_state is None:
        return np.random.default_rng()
    elif isinstance(random_state, int):
        return np.random.default_rng(random_state)
    elif isinstance(random_state, np.random.Generator):
        return random_state
    else:
        raise ValueError(
            "random_state must be an int or a numpy random Generator instance."
            f" Got {type(random_state)}."
        )


def _check_distributions(
    distributions: (
        scipy.stats.rv_continuous
        | scipy.stats.rv_discrete
        | list[scipy.stats.rv_continuous]
        | list[scipy.stats.rv_discrete]
    ),
) -> tuple[list[scipy.stats.rv_continuous | scipy.stats.rv_discrete], int, np.dtype]:
    """Check if distributions are valid and return as a list.

    Parameters
    ----------
    distributions : list of `scipy.stats.rv_continuous` or `scipy.stats.rv_discrete`
        List of distributions for each segment.

    Returns
    -------
    list[scipy.stats.rv_continuous | scipy.stats.rv_discrete]
        List of distributions for each segment, where each distribution is guarnteed
        to have a `rvs(size: int, random_state: int | None)` method that returns
        a numpy array or scalar of the same size, and the output size is either
        1 or `p`.
    int
        Output size of the distributions, which is either 1 or `p`.
    """
    if not isinstance(distributions, list):
        distributions = [distributions]

    output_sizes = []
    output_dtypes = []
    for dist in distributions:
        try:
            output = dist.rvs(size=1)
            output_sizes.append(output.size)
            output_dtypes.append(output.dtype)
        except Exception:
            output_sizes.append(None)

    if any(size is None for size in output_sizes):
        raise ValueError(
            "All distributions must support the 'rvs' method with a 'size' argument,"
            " where the output is a numpy.array or numpy scalar."
            " Ensure that all distributions are valid scipy.stats distributions."
        )

    if len(set(output_sizes)) > 1:
        raise ValueError(
            f"All distributions must produce samples with the same number of variables."
            f" Got distribution.rvs(size=1).size outputs: {output_sizes}."
        )

    if len(set(output_dtypes)) > 1:
        raise ValueError(
            "All distributions must produce samples with the same data type."
            f" Got distribution.rvs(size=1).dtype outputs: {output_dtypes}."
        )

    return distributions, output_sizes[0], output_dtypes[0]


def generate_piecewise_data(
    distributions: scipy.stats.rv_continuous
    | scipy.stats.rv_discrete
    | list[scipy.stats.rv_continuous]
    | list[scipy.stats.rv_discrete],
    lengths: int | list[int] | np.ndarray | None = None,
    n_samples: int = 100,
    random_state: int | np.random.Generator | None = None,
    return_params: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Generate data with a piecewise constant distribution.

    Parameters
    ----------
    distributions : list of `scipy.stats.rv_continuous` or `scipy.stats.rv_discrete`
        The distribution for each segment. Each distribution is expected
        to be a scipy distribution object (e.g., `scipy.stats.norm`,
        `scipy.stats.uniform`). See
        `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_
        for a list of all available distributions. However, the function will run as
        long as the distribution objects support an
        `rvs(size: int, random_state: int | None)` method.
    lengths : int, list of int or np.ndarray, optional (default=None)
        Lengths of each segment. If a list or array, it must be of the same
        length as `distributions`. If an integer is provided, all segments will be
        of this length.
    n_samples : int (default=100)
        Total number of samples to generate if `lengths` is not specified.
        In this case, `lengths` are randomly generated in the following way:

        1. `len(distributions) - 1` change points are sampled uniformly from the range
           `1:n_samples` without replacement.
        2. `lengths` are then computed as the differences between these change points.

    random_state : int, optional
        Seed for the random number generator. The random state per distribution is set
        to random_state + i, where i is the index of the distribution in the list.
    return_params : bool, optional (default=False)
        If True, the function returns a tuple of the generated DataFrame and a
        dictionary with the parameters used to generate the data.

    Returns
    -------
    pd.DataFrame
        Data frame with generated data.

    dict, optional
        A dictionary containing the parameters used to generate the data. Only returned
        if `return_params` is True. It has the following keys:

        * `"distributions"` : list of `scipy.stats.rv_continuous` or
          `scipy.stats.rv_discrete` with the distributions used for each segment.
        * `"lengths"` : list of lengths for each segment.
        * `"change_points"` : list of change points, which are the starting indices
          of each segment in the data.
        * `"n_samples"` : total number of samples generated.

    Examples
    --------
    >>> # Example 1: Two normal segments
    >>> from skchange.datasets import generate_piecewise_data
    >>> from scipy.stats import norm
    >>> generate_piecewise_data(
    ...     distributions=[norm(0, 1), norm(10, 0.1)],
    ...     lengths=[7, 3],
    ...     random_state=1,
    ... )
               0
    0   0.345584
    1   0.821618
    2   0.330437
    3  -1.303157
    4   0.905356
    5   0.446375
    6  -0.536953
    7  10.058112
    8  10.036457
    9  10.029413

    >>> # Example 2: Two Poisson segments
    >>> from scipy.stats import poisson
    >>> generate_piecewise_data(
    ...     distributions=[poisson(1), poisson(10)],
    ...     lengths=[5, 5],
    ...     random_state=2,
    ... )
        0
    0   0
    1   0
    2   1
    3   2
    4   0
    5   8
    6  11
    7   9
    8   9
    9   9


    >>> # Example 3: Specify only n_samples and distributions, random segment lengths
    >>> generate_piecewise_data(
    ...     distributions=[norm(0), norm(5)],
    ...     n_samples=8,
    ...     random_state=3,
    ... )
              0
    0 -2.555665
    1  0.418099
    2 -0.567770
    3 -0.452649
    4 -0.215597
    5 -2.019986
    6  4.768068
    7  4.134787
    """
    random_state = check_random_state(random_state)

    distributions, n_variables, dtype = _check_distributions(distributions)
    n_segments = len(distributions)

    if lengths is None:
        if n_samples < n_segments:
            raise ValueError("total_samples must be at least equal to `n_segments`.")
        if n_segments == 1:
            lengths = [n_samples]
        else:
            change_points = random_state.choice(
                np.arange(1, n_samples), size=n_segments - 1, replace=False
            )
            lengths = np.diff(
                np.concatenate(([0], np.sort(change_points), [n_samples]))
            )
    elif isinstance(lengths, Number):
        lengths = [lengths] * n_segments

    if len(distributions) != len(lengths):
        raise ValueError("Length of `distributions` and `lengths` must match.")

    if any([length <= 0 for length in lengths]):
        raise ValueError("All lengths must be positive integers.")

    ends = np.cumsum(lengths)
    starts = np.concatenate(([0], ends[:-1]))
    _n_samples = np.sum(lengths)
    generated_values = np.empty((_n_samples, n_variables), dtype=dtype)
    for distribution, start, end in zip(distributions, starts, ends):
        length = end - start
        values = distribution.rvs(size=length, random_state=random_state)
        generated_values[start:end, :] = values.reshape(length, n_variables)

    generated_df = pd.DataFrame(generated_values)

    if return_params:
        params = {
            "distributions": distributions,
            "lengths": lengths,
            "change_points": starts[1:],
            "n_samples": _n_samples,
        }
        return generated_df, params

    return generated_df
