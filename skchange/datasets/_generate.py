"""Data generators."""

__author__ = ["Tveten"]

from numbers import Number

import numpy as np
import pandas as pd
import scipy.stats


def _check_random_state(
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


def _check_change_points(change_points: int | list[int], n: int) -> list[int]:
    """Check if change points are valid.

    Parameters
    ----------
    change_points : list of int
        List of change points.
    n : int
        Total number of observations.

    Raises
    ------
    ValueError
        If any change point is out of bounds.
    """
    if isinstance(change_points, Number):
        change_points = [change_points]

    change_points = sorted(change_points)
    if any([cpt > n - 1 or cpt < 0 for cpt in change_points]):
        raise ValueError(
            "Change points must be within the range of the data."
            f" Got n={n}, max(change_points)={change_points} and"
            f" min(change_points)={min(change_points)}."
        )
    if len(change_points) >= 2 and min(np.diff(change_points)) < 1:
        raise ValueError(
            "Change points must be at least 1 apart."
            f" Got change_points={change_points}."
        )

    return change_points


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


def _check_proportion_affected(
    proportion_affected: list[float] | np.ndarray,
    n_segments: int,
) -> None:
    """Check if the proportion of affected variables is valid.

    Parameters
    ----------
    proportion_affected : list[float] | np.ndarray
        Proportion of affected variables.
    n_segments : int
        Number of segments.

    Returns
    -------
    list[float]
        List of validated proportions for each segment.
    """
    if len(proportion_affected) != n_segments:
        raise ValueError(
            "Number of `proportions_affected` must match the number of segments."
            f" Got {len(proportion_affected)} proportions and {n_segments} segments."
        )

    for prop in proportion_affected:
        if not (0 < prop <= 1):
            raise ValueError(
                "Proportion of affected variables must be between (0, 1]."
                f" Got `proportion_affected`={prop}."
            )


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
        List of distributions for each segment, where each distribution is expected
        to be a scipy distribution object (e.g., `scipy.stats.norm`,
        `scipy.stats.uniform`). See
        `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_
        for a list of all available distributions. However, the function will run as
        long as the distribution objects support an
        `rvs(size: int, random_state: int | None)` method.
    lengths : int, list of int or np.ndarray, optional (default=None)
        List of lengths for each segment. If a single integer is provided,
        it will be used for all segments.
    n_samples : int (default=100)
        Total number of samples to generate if `lengths` is not specified.
        The lengths of the segments are randomly generated by sampling
        `len(distributions) - 1` change points uniformly from the range `1:n_samples`
        without replacement.
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
    random_state = _check_random_state(random_state)

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


def _get_n_segments(
    means: float | np.ndarray | list[float] | list[np.ndarray] | None,
    variances: float | np.ndarray | list[float] | list[np.ndarray] | None,
    lengths: int | list[int] | np.ndarray | None,
    n_segments: int | None,
    n_samples: int,
    random_state: int | np.random.Generator | None = None,
) -> int:
    if isinstance(means, list):
        if len(means) == 0:
            raise ValueError(
                "If `means` is a list, it must contain at least one element."
            )
        return len(means)
    if isinstance(variances, list):
        if len(variances) == 0:
            raise ValueError(
                "If `variances` is a list, it must contain at least one element."
            )
        return len(variances)
    if isinstance(lengths, (list, np.ndarray)):
        if len(lengths) == 0:
            raise ValueError(
                "If `lengths` is a list or array, it must contain at least one element."
            )
        return len(lengths)

    if n_segments is None:
        mean_n_cpts = 4
        binom_prob = min(0.1, mean_n_cpts / n_samples)
        n_change_points = scipy.stats.binom(n_samples, binom_prob).rvs(
            size=1, random_state=random_state
        )[0]
        n_segments = n_change_points + 1
    else:
        if n_segments < 1:
            raise ValueError("Number of segments must be at least 1.")

    return int(n_segments)


def _get_affected_variables(
    proportion_affected: float | list[float] | np.ndarray | None,
    randomise_affected_variables: bool,
    n_segments: int,
    n_variables: int,
    random_state: int | np.random.Generator | None = None,
) -> list[np.ndarray]:
    random_state = _check_random_state(random_state)

    if proportion_affected is None:
        proportion_affected = scipy.stats.uniform(1e-8, 1).rvs(
            size=n_segments, random_state=random_state
        )
    elif isinstance(proportion_affected, Number):
        proportion_affected = [proportion_affected] * n_segments

    _check_proportion_affected(proportion_affected, n_segments)

    affected_variables = []
    for prop in proportion_affected:
        affected_size = int(np.ceil(n_variables * prop))
        if randomise_affected_variables:
            affected_vars = np.sort(
                random_state.choice(n_variables, size=affected_size, replace=False)
            )
        else:
            affected_vars = np.arange(affected_size)
        affected_variables.append(affected_vars)

    return affected_variables


def _get_means(
    means: float | np.ndarray | list[float] | list[np.ndarray] | None,
    n_segments: int,
    n_variables: int,
    affected_variables: list[np.ndarray],
    random_state: int | np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Create means for each segment."""
    if means is None or isinstance(means, (Number, np.ndarray)):
        means = [means] * n_segments

    if len(means) != n_segments:
        raise ValueError(
            "Number of means must match number of segments."
            f" Got {len(means)} means and {n_segments} segments."
        )

    _means = [np.zeros(n_variables)]  # Initialize for the loop to work. Remove later.
    for mean, affected in zip(means, affected_variables):
        prev_mean = _means[-1].copy()
        if mean is None:
            # The affected set are the variables that change, so the next mean vector
            # should be the same, except for the affected variables.
            _mean = prev_mean
            _mean[affected] = scipy.stats.norm(0, 2).rvs(
                size=affected.size, random_state=random_state
            )
        elif isinstance(mean, Number):
            _mean = prev_mean
            _mean[affected] = mean
        else:
            _mean = np.asarray(mean).reshape(-1)

        if _mean.shape[0] != n_variables:
            raise ValueError(
                "Mean vector must have the same length as the number of variables."
                f" Got mean={_mean} with shape {_mean.shape}"
                f" and n_variables={n_variables}."
            )
        _means.append(_mean)

    return _means[1:]  # The first element is just a placeholder to initialize the loop.


def _get_covs(
    covs: float | np.ndarray | list[float] | list[np.ndarray] | None,
    n_segments: int,
    n_variables: int,
    affected_variables: list[np.ndarray],
    random_state: int | np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Create covariance matrices for each segment."""
    if covs is None or isinstance(covs, (Number, np.ndarray)):
        covs = [covs] * n_segments

    if len(covs) != n_segments:
        raise ValueError(
            "Number of variances must match number of segments."
            f" Got {len(covs)} variances and {n_segments} segments."
        )

    _vars = [np.ones(n_variables)]  # Initialize for the loop to work. Remove later.
    _covs = []
    for cov, affected in zip(covs, affected_variables):
        prev_var = _vars[-1].copy()
        if cov is None:
            # The affected set are the variables that change, so the next covariance
            # matrix should be the same, except for the affected variables.
            _var = prev_var
            _var[affected] = scipy.stats.chi2(2).rvs(
                size=affected.size, random_state=random_state
            )
        elif isinstance(cov, Number):
            _var = prev_var
            _var[affected] = cov
        else:
            _var = np.asarray(cov)

        if _var.ndim == 1:
            _cov = np.diag(_var)
        elif _var.ndim == 2 and _var.shape[0] == _var.shape[1]:
            _cov = _var
        else:
            raise ValueError(
                "Covariance matrix must be a square matrix with shape (p, p)."
                f" Got covariance matrix with shape {_cov.shape} and p={n_variables}."
            )

        eigvals = np.linalg.eigvalsh(_cov)
        if np.any(eigvals <= 0):
            raise ValueError("Covariance matrix must be positive definite.")

        _covs.append(_cov)

    return _covs


def generate_piecewise_normal_data(
    means: float | np.ndarray | list[float] | list[np.ndarray] | None = None,
    variances: float | np.ndarray | list[float] | list[np.ndarray] | None = 1.0,
    lengths: int | list[int] | np.ndarray | None = None,
    n_samples: int = 100,
    n_segments: int | None = None,
    n_variables: int = 1,
    proportion_affected: float | list[float] | np.ndarray | None = None,
    randomise_affected_variables: bool = False,
    random_state: int = None,
    return_params: bool = False,
) -> pd.DataFrame:
    """Generate piecewise multivariate normal data.

    Parameters
    ----------
    means : float or list of float or list of np.ndarray, optional (default=None)
        Means for each segment. If None, random means are generated according to a
        normal distribution with mean 0 and standard deviation 2. If specified and
        `n_change_points` or `change_points` is not provided, the number of means or
        govern the number of segments.
    variances : float or list of float or list of np.ndarray, optional (default=1.0)
        Variances or covariance matrices for each segment. Vectors are treated as
        diagonal covariance matrices. If None, random variance vectors are generated
        according to a chi-squared distribution with 2 degrees of freedom.
        If specified and `n_change_points` or `change_points` is not provided, the
        number of variances or covariance matrices govern the number of segments.
    lengths : int, list of int or np.ndarray, optional (default=None)
        List of lengths for each segment. If a single integer is provided,
        it will be used for all segments.
    n_samples : int (default=100)
        Total number of samples to generate if `lengths` is not specified.
        The lengths of the segments are randomly generated by sampling
        `len(distributions) - 1` change points uniformly from the range `1:n_samples`
        without replacement. The lengths are given by the difference between
        consecutive change points.
    n_segments : int, optional (default=None)
        Number of segments to generate if neither `means`, `variances`, nor
        `lengths` are specified. If None, the number of segments is randomly generated
    n_variables : int, optional (default=1)
        Number of variables.
    proportion_affected: float, optional (default=1.0)
        Proportion of variables 1, ..., `n_variables` that are affected by each change.
        This is used to determine how many variables are affected by the change in
        means and variances. By default all variables are affected. Only relevant for
        unspecified `means` or `variances`.
    random_state : int, optional
        Seed for the random number generator. The random_state is used as a basis for
        random generation of all random entities, including the change points, means and
        variances (if specified). If None, the random state is not set.
    return_params: bool, optional (default=False)
        If True, the function returns a tuple of the generated DataFrame and a
        dictionary with the parameters used to generate the data, including
        `change_points`, `means`, and `variances`. If False, only the DataFrame is
        returned.

    Returns
    -------
    pd.DataFrame
        DataFrame with generated data. The DataFrame has `n` rows and `p` columns.

    dict
        A dictionary containing the parameters used to generate the data. It has
        keys `change_points`, `means`, and `variances`. It is returned only if
        `return_params` is True.

    Examples
    --------
    >>> from skchange.datasets import generate_piecewise_normal_data
    >>> df = generate_piecewise_normal_data(n=10, change_points = [5], means=[0, 5])
    >>> df
              0
    0  0.614884
    1  0.266904
    2  0.812941
    3  1.300181
    4 -0.506429
    5  6.441842
    6  6.069931
    7  5.449876
    8  3.924228
    9  5.514109
    """
    random_state = _check_random_state(random_state)
    n_segments = _get_n_segments(
        means,
        variances,
        lengths,
        n_segments,
        n_samples,
        random_state,
    )
    affected_variables = _get_affected_variables(
        proportion_affected,
        randomise_affected_variables,
        n_segments,
        n_variables,
        random_state,
    )
    means = _get_means(
        means,
        n_segments,
        n_variables,
        affected_variables,
        random_state,
    )
    covs = _get_covs(
        variances,
        n_segments,
        n_variables,
        affected_variables,
        random_state,
    )
    distributions = [
        scipy.stats.multivariate_normal(mean=mean, cov=cov)
        for mean, cov in zip(means, covs)
    ]
    df, _params = generate_piecewise_data(
        distributions,
        lengths,
        n_samples,
        random_state,
        return_params=True,
    )
    if return_params:
        params = {
            "n_segments": n_segments,
            "n_samples": _params["n_samples"],
            "means": means,
            "variances": variances,
            "lengths": _params["lengths"],
            "change_points": _params["change_points"],
            "affected_variables": affected_variables,
        }
        return df, params
    else:
        return df


def generate_changing_data(
    n: int = 100,
    changepoints: int | list[int] = 50,
    means: float | list[float] | list[np.ndarray] = 0.0,
    variances: float | list[float] | list[np.ndarray] = 1.0,
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

    means = [np.asarray(mean).reshape(-1) for mean in means]
    variances = [np.asarray(variance).reshape(-1) for variance in variances]

    n_segments = len(changepoints) + 1
    if len(means) == 1:
        means = means * n_segments
    if len(variances) == 1:
        variances = variances * n_segments

    if n_segments != len(means) or n_segments != len(variances):
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
    x = scipy.stats.multivariate_normal.rvs(np.zeros(p), np.eye(p), n, random_state)
    changepoints = [0] + changepoints + [n]
    for prev_cpt, next_cpt, mean, variance in zip(
        changepoints[:-1], changepoints[1:], means, variances
    ):
        x[prev_cpt:next_cpt] = mean + np.sqrt(variance) * x[prev_cpt:next_cpt]

    out_columns = [f"var{i}" for i in range(p)]
    df = pd.DataFrame(x, index=range(len(x)), columns=out_columns)
    return df


def generate_anomalous_data(
    n: int = 100,
    anomalies: tuple[int, int] | list[tuple[int, int]] = (70, 80),
    means: float | list[float] | list[np.ndarray] = 3.0,
    variances: float | list[float] | list[np.ndarray] = 1.0,
    random_state: int = None,
) -> pd.DataFrame:
    """
    Generate multivariate normal data with anomalies.

    Parameters
    ----------
    n : int, optional (default=100)
        Number of observations.
    anomalies : list of tuples, optional (default=[(71, 80)])
        List of tuples of the form [start, end) indicating the start and end of an
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

    means = [np.asarray(mean).reshape(-1) for mean in means]
    variances = [np.asarray(variance).reshape(-1) for variance in variances]

    if len(means) == 1:
        means = means * len(anomalies)
    if len(variances) == 1:
        variances = variances * len(anomalies)

    if len(anomalies) != len(means) or len(anomalies) != len(variances):
        raise ValueError("Number of anomalies, means and variances must be the same.")
    if any([len(anomaly) != 2 for anomaly in anomalies]):
        raise ValueError("Anomalies must be of length 2.")
    if any([anomaly[1] <= anomaly[0] for anomaly in anomalies]):
        raise ValueError("The start of an anomaly must be before its end.")
    if any([anomaly[1] > n for anomaly in anomalies]):
        raise ValueError("Anomalies must be within the range of the data.")

    p = len(means[0])
    x = scipy.stats.multivariate_normal.rvs(np.zeros(p), np.eye(p), n, random_state)
    for anomaly, mean, variance in zip(anomalies, means, variances):
        start, end = anomaly
        x[start:end] = mean + np.sqrt(variance) * x[start:end]

    out_columns = [f"var{i}" for i in range(p)]
    df = pd.DataFrame(x, index=range(len(x)), columns=out_columns)
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
    changepoints = [segment_length * i for i in range(1, n_segments)]
    return generate_changing_data(n, changepoints, means, vars, random_state)


def generate_continuous_piecewise_linear_signal(
    change_points, slopes, intercept=0, n_samples=200, noise_std=0.1, random_seed=None
):
    """Generate a continuous piecewise linear signal with noise.

    Parameters
    ----------
    change_points : list
        List of indices where the slope changes (kink points)
    slopes : list
        List of slopes for each segment (should be one more than change_points)
    intercept : float, default=0
        Starting intercept value
    n_samples : int, default=200
        Total number of samples
    noise_std : float, default=0.1
        Standard deviation of the Gaussian noise to add
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with the signal and corresponding time points
    list
        List of true change points (as indices)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if len(slopes) != len(change_points) + 1:
        raise ValueError(  # pragma: no cover
            "Number of slopes must be one more than number of change points"
        )

    # Create time points and allocate signal
    time = np.arange(n_samples)
    signal = np.zeros(n_samples)

    # First segment
    signal[: change_points[0]] = intercept + slopes[0] * time[: change_points[0]]
    current_value = signal[change_points[0] - 1]

    # Middle segments
    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i + 1]
        segment_time = time[start_idx:end_idx] - time[start_idx]
        signal[start_idx:end_idx] = current_value + slopes[i + 1] * segment_time
        current_value = signal[end_idx - 1]

    # Last segment
    if len(change_points) > 0:
        last_start = change_points[-1]
        segment_time = time[last_start:] - time[last_start]
        signal[last_start:] = current_value + slopes[-1] * segment_time

    # Add noise
    signal += np.random.normal(0, noise_std, n_samples)

    # Convert to DataFrame
    df = pd.DataFrame({"signal": signal})

    return df
