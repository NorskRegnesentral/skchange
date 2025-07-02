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


def _check_means(
    means: float | np.ndarray | list[float] | list[np.ndarray],
    p: int,
    change_points: list[int],
) -> list[np.ndarray]:
    """Check if means are valid and convert to numpy arrays.

    Parameters
    ----------
    means : float or list of float or list of np.ndarray
        Means for each segment.
    p : int
        Number of variables in the data.
    change_points : list[int]
        List of change points.

    Returns
    -------
    list[np.ndarray]
        List of means as numpy arrays.
    """
    n_segments = len(change_points) + 1
    if isinstance(means, (Number, np.ndarray)):
        means = [means] * n_segments

    _means = []
    for mean in means:
        if isinstance(mean, Number):
            _mean = np.full(p, mean)
        else:
            _mean = np.asarray(mean).reshape(-1)

        if _mean.shape[0] != p:
            raise ValueError(
                "Mean vector must have the same length as the number of variables p."
                f" Got mean={_mean} with shape {_mean.shape} and p={p}."
            )
        _means.append(_mean)

    if len(_means) != n_segments:
        raise ValueError(
            "Number of means must match number of segments."
            f" Got {len(_means)} means and {n_segments} segments."
        )

    return _means


def _check_variances(
    variances: float | np.ndarray | list[float] | list[np.ndarray],
    p: int,
    change_points: list[int],
) -> list[np.ndarray]:
    """Check if variances are valid and return as 2D covariance matrices.

    Parameters
    ----------
    variances : float or list of float or list of np.ndarray
        Variances or covariance matrices for each segment.
    p : int
        Number of variables in the data.
    change_points : list[int]
        List of change points.

    Returns
    -------
    list[np.ndarray]
        List of covariance matrices as 2D numpy arrays.
    """
    n_segments = len(change_points) + 1
    if isinstance(variances, (Number, np.ndarray)):
        variances = [variances] * n_segments

    covariances = []
    for variance in variances:
        if isinstance(variance, Number):
            _variance = np.full(p, variance)
        else:
            _variance = np.asarray(variance)

        if _variance.ndim == 1:
            cov = np.diag(_variance)
        elif _variance.ndim == 2:
            if _variance.shape[0] != _variance.shape[1]:
                raise ValueError("Covariance matrix must be square.")
            cov = _variance
        else:
            raise ValueError("Variance input must be scalar, 1D, or 2D array.")

        if cov.shape[0] != p or cov.shape[1] != p:
            raise ValueError(
                "Covariance matrix must have the shape (p, p)."
                f" Got covariance matrix with shape {cov.shape} and p={p}."
            )

        if not np.allclose(cov, cov.T, atol=1e-8):
            raise ValueError("Covariance matrix must be symmetric.")

        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals <= 0):
            raise ValueError("Covariance matrix must be positive definite.")

        covariances.append(cov)

    if len(variances) != n_segments:
        raise ValueError(
            "Number of variances must match number of segments."
            f" Got {len(variances)} variances and {n_segments} segments."
        )

    return covariances


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
    proportion_affected: float | list[float] | np.ndarray,
    change_points: list[int],
) -> list[float]:
    """Check if the proportion of affected variables is valid.

    Parameters
    ----------
    proportion_affected : float | list[float] | np.ndarray
        Proportion of affected variables.
    change_points : list[int]
        List of change points.

    Returns
    -------
    list[float]
        List of validated proportions for each segment.
    """
    n_segments = len(change_points) + 1
    if isinstance(proportion_affected, Number):
        proportion_affected = [proportion_affected] * n_segments

    if len(proportion_affected) != n_segments:
        raise ValueError(
            "Number of proportions must match number of segments."
            f" Got {len(proportion_affected)} proportions and {n_segments} segments."
        )

    for prop in proportion_affected:
        if not (0 < prop <= 1):
            raise ValueError(
                "Proportion of affected variables must be between (0, 1]."
                f" Got proportion_affected={prop}."
            )

    return proportion_affected


def _check_affected_variables(
    affected_variables: np.ndarray | list[np.ndarray | list[int]],
    p: int,
    change_points: list[int],
) -> list[np.ndarray]:
    """Check if affected variables are valid.

    Parameters
    ----------
    affected_variables : list of np.ndarray
        List of arrays with indices of affected variables for each segment.
    p : int
        Number of variables in the data.
    change_points : list[int]
        List of change points.

    Returns
    -------
    list[np.ndarray]
        List of validated affected variable indices for each segment.
    """
    n_segments = len(change_points) + 1

    if not isinstance(affected_variables, list):
        affected_variables = [affected_variables] * n_segments

    if len(affected_variables) != n_segments:
        raise ValueError(
            "Number of affected variable arrays must match number of segments."
            f" Got {len(affected_variables)} arrays and {n_segments} segments."
        )

    for i, affected in enumerate(affected_variables):
        if not isinstance(affected, (np.ndarray, list)):
            raise ValueError(
                f"Affected variables for segment {i} must be a numpy array or list."
                f" Got {type(affected)}."
            )
        if np.any(affected < 0) or np.any(affected >= p):
            raise ValueError(
                f"Affected variable indices for segment {i} must be in the range"
                f" [0, {p})."
                f" Got affected={affected}."
            )
        if len(set(affected)) != len(affected):
            raise ValueError(
                f"Affected variable indices for segment {i} must be unique."
                f" Got affected={affected}."
            )

    return affected_variables


def _generate_random_1d_array(
    p: int,
    baseline_value: float,
    affected_variables: np.ndarray | list[int],
    distribution: scipy.stats.rv_continuous,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate a random 1D array.

    Parameters
    ----------
    p : int
        Number of variables.
    baseline_value : float, optional (default=0.0)
        Value to fill the array with for unaffected variables.
    affected_variables : np.ndarray or list[int] or None, optional
        Indices of variables that have random values generated by `distribution`.
        If None, all variables are affected.
    distribution : `scipy.stats.rv_continuous`, optional
        Distribution to sample from for the affected variables. If None, a standard
        normal distribution is used.
    random_state : int or None, optional
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        Vector of shape (p,).
    """
    out_array = np.full(p, baseline_value, dtype=float)
    out_array[affected_variables] = distribution.rvs(
        size=len(affected_variables), random_state=random_state
    )
    return out_array


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
        return generated_df, {
            "distributions": distributions,
            "lengths": lengths,
            "change_points": starts[1:],
            "n_samples": _n_samples,
        }

    return generated_df


def generate_piecewise_normal_data(
    n: int = 100,
    p: int = 1,
    n_change_points: int | None = None,
    change_points: int | list[int] | np.ndarray | None = None,
    means: float | np.ndarray | list[float] | list[np.ndarray] | None = None,
    variances: float | np.ndarray | list[float] | list[np.ndarray] | None = 1.0,
    proportion_affected: float = 1.0,
    random_state: int = None,
    return_params: bool = False,
) -> pd.DataFrame:
    """Generate piecewise multivariate normal data.

    Parameters
    ----------
    n : int, optional (default=100)
        Number of samples.
    p : int, optional (default=1)
        Number of variables.
    n_change_points : int, optional (default=None)
        Number of change points to generate. Ignored if `change_points` is provided.
        If None, the number of change points is
        randomly generated from a binomial distribution with parameters `n` and
        `p=min(0.5, 5 / n)`, which gives 5 change points on average.
    change_points : int or list of int or np.ndarray, optional (default=None)
        Change points in the data. If None, `n_change_points` change points locations
        are randomly drawn from the range `[1, n-1]` without replacement.
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
    proportion_affected: float, optional (default=1.0)
        Proportion of variables 1, ..., p that are affected by each change. This is
        used to determine how many variables are affected by the change in means and
        variances. By default all variables are affected. Only relevant for unspecified
        `means` or `variances`.
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
    if n < 1:
        raise ValueError(f"Number of samples n must be at least 1. Got n={n}.")
    if p < 1:
        raise ValueError(f"Number of variables p must be at least 1. Got p={p}.")

    if (
        isinstance(means, list)
        and isinstance(variances, list)
        and len(means) != len(variances)
    ):
        raise ValueError(
            f"Number of means and variances must match."
            f" Got {len(means)} means and {len(variances)} variances."
        )

    if change_points is not None:
        n_change_points = 1 if isinstance(change_points, Number) else len(change_points)

    if (means is not None or variances is not None) and n_change_points is None:
        n_change_points = max(
            len(means) - 1 if isinstance(means, list) else 0,
            len(variances) - 1 if isinstance(variances, list) else 0,
        )

    if n_change_points is None:
        mean_n_cpts = 5
        binom_prob = min(0.5, mean_n_cpts / n)
        n_change_points = scipy.stats.binom(n, binom_prob).rvs(
            size=1, random_state=random_state
        )

    if n_change_points < 0:
        raise ValueError("Number of change points must be non-negative.")

    if change_points is None:
        change_points = np.random.default_rng(random_state).choice(
            np.arange(1, n), size=n_change_points, replace=False
        )
    change_points = _check_change_points(change_points, n)
    n_segments = len(change_points) + 1

    if proportion_affected <= 0 or proportion_affected > 1:
        raise ValueError(
            "Proportion of affected variables must be between (0, 1]."
            f" Got proportion_affected={proportion_affected}."
        )
    n_affected = int(np.ceil(p * proportion_affected))
    affected_variables = [
        np.random.default_rng(
            random_state + 1 if random_state is not None else None
        ).choice(p, size=n_affected, replace=False)
        for _ in range(n_segments)
    ]

    if means is None:
        means = [
            # Change the random state for each segment to ensure different means
            # std = 2 to get a reasonable spread of means.
            _generate_random_1d_array(
                p=p,
                baseline_value=0.0,
                affected_variables=affected_variables[i],
                distribution=scipy.stats.norm(0, 2),
                random_state=random_state + i if random_state is not None else None,
            )
            for i in range(n_segments)
        ]
    means = _check_means(means, p, change_points)

    if variances is None:
        variances = [
            # Change the random state for each segment to ensure different variances.
            # chi2(2) is used to get a reasonable spread of variances.
            _generate_random_1d_array(
                p=p,
                baseline_value=1.0,
                affected_variables=affected_variables[i],
                distribution=scipy.stats.chi2(2),
                random_state=random_state + i if random_state is not None else None,
            )
            for i in range(n_segments)
        ]
    variances = _check_variances(variances, p, change_points)

    distributions = [
        scipy.stats.multivariate_normal(mean=mean, cov=cov)
        for mean, cov in zip(means, variances)
    ]
    lengths = np.diff(np.concatenate(([0], change_points, [n]))).astype(int)
    df = generate_piecewise_data(
        distributions=distributions,
        lengths=lengths,
        random_state=random_state,
    )

    if return_params:
        params = {
            "change_points": change_points,
            "means": means,
            "variances": variances,
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


def _segment_label_matrix_from_change_points(
    n: int,
    p: int,
    change_points: np.ndarray | list[int],
    affected_variables: list[np.ndarray],
):
    """Create a segment label matrix from change points."""
    segment_labels = np.zeros((n, p), dtype=int)
    for i, (cpt, affected) in enumerate(zip(change_points, affected_variables)):
        segment_labels[cpt:, affected] = i + 1
    return segment_labels


def _segment_label_matrix_from_anomalies(
    n: int,
    p: int,
    anomalies: np.ndarray | list[tuple[int, int]],
    affected_variables: list[np.ndarray],
):
    """Create a segment label matrix from anomalies."""
    segment_labels = np.zeros((n, p), dtype=int)
    for i, ((start, end), affected) in enumerate(zip(anomalies, affected_variables)):
        segment_labels[start:end, affected] = i + 1
    return segment_labels


def generate_piecewise_data_new(
    n: int = 100,
    p: int = 1,
    distributions: (
        scipy.stats.rv_continuous
        | scipy.stats.rv_discrete
        | list[scipy.stats.rv_continuous]
        | list[scipy.stats.rv_discrete]
        | None
    ) = None,
    change_points: int | list[int] | np.ndarray | None = None,
    proportion_affected: float | list[float] | np.ndarray | None = None,
    affected_variables: np.ndarray | list[np.ndarray | list[int]] | None = None,
    random_state: int | np.random.Generator | None = None,
    return_params: bool = False,
) -> pd.DataFrame:
    """Generate piecewise data.

    Parameters
    ----------
    n : int, optional (default=100)
        Number of samples.
    p : int, optional (default=1)
        Number of variables.
    n_change_points : int, optional (default=None)
        Number of change points to generate. Ignored if `change_points` or
        `distributions` is provided. If None, the number of change points is
        randomly generated from a binomial distribution with parameters `n` and
        `p=min(0.5, 5 / n)`, which gives 5 change points on average.
    change_points : int or list of int or np.ndarray, optional (default=None)
        Change points in the data. If None, `n_change_points` change points locations
        are randomly drawn from the range `[1, n-1]` without replacement.
    distributions : list of `scipy.stats.rv_continuous` or `scipy.stats.rv_discrete`, optional (default=None)
        List of distributions for each segment, where each distribution is expected
        to be a scipy distribution object (e.g., `scipy.stats.norm`,
        `scipy.stats.uniform`). See `scipy.stats
        <https://docs.scipy.org/doc/scipy/reference/stats.html>`_
        for a list of all available distributions. However, the function will run as
        long as the distribution objects support an
        `rvs(size: int, random_state: int | None)` method.
    proportion_affected: float, optional (default=1.0)
        Proportion of variables 1, ..., p that are affected by each change. This is
        used to determine how many variables are affected by the change in means and
        variances. By default all variables are affected. Only relevant for unspecified
        `means` or `variances`.
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
    """  # noqa: E501
    random_state = _check_random_state(random_state)

    if n < 1:
        raise ValueError(f"Number of samples n must be at least 1. Got n={n}.")
    if p < 1:
        raise ValueError(f"Number of variables p must be at least 1. Got p={p}.")

    if change_points is not None:
        n_change_points = 1 if isinstance(change_points, Number) else len(change_points)
    if distributions is not None and n_change_points is None:
        n_change_points = (
            len(distributions) - 1 if isinstance(distributions, list) else 0
        )
    if n_change_points is None:
        mean_n_cpts = 5
        binom_prob = min(0.1, mean_n_cpts / n)
        n_change_points = scipy.stats.binom(n, binom_prob).rvs(
            size=1, random_state=random_state
        )
    if n_change_points < 0:
        raise ValueError("Number of change points must be non-negative.")

    if change_points is None:
        change_points = random_state.choice(
            np.arange(1, n), size=n_change_points, replace=False
        )
    change_points = _check_change_points(change_points, n)
    n_segments = len(change_points) + 1

    if distributions is None:
        means = scipy.stats.norm(0, 2).rvs(n_segments, random_state=random_state)
        distributions = [scipy.stats.norm(mean) for mean in means]

    # n_variables_distribution is guaranteed to be 1 or p.
    distributions, n_variables_distribution, dtype_distribution = _check_distributions(
        distributions, p, change_points
    )

    if affected_variables is None:
        if proportion_affected is None:
            proportion_affected = scipy.stats.uniform(1e-8, 1).rvs(
                size=n_segments, random_state=random_state
            )
        proportion_affected = _check_proportion_affected(
            proportion_affected, change_points
        )
        affected_variables = [
            np.sort(random_state.choice(p, size=int(np.ceil(p * prop)), replace=False))
            for prop in proportion_affected
        ]
    affected_variables = _check_affected_variables(affected_variables, p, change_points)

    generated_data = np.empty((n, p), dtype=dtype_distribution)
    if n_variables_distribution == p:
        starts = np.concatenate(([0], change_points))
        ends = np.concatenate((change_points, [n]))
        for start, end, distribution in zip(starts, ends, distributions):
            generated_data[start:end, :] = distribution.rvs(
                size=end - start, random_state=random_state
            )
        # affected_variables not used in this case. None will be returned in the params
        # dictionary if return_params is True.
        affected_variables = None
    else:
        labels = _segment_label_matrix_from_change_points(
            n=n,
            p=p,
            change_points=change_points,
            affected_variables=affected_variables,
        )
        for i, distribution in enumerate(distributions):
            label_mask = labels == i
            generated_data[label_mask] = distribution.rvs(
                size=np.sum(label_mask), random_state=random_state
            )

    df = pd.DataFrame(generated_data)

    if return_params:
        params = {
            "change_points": change_points,
            "distributions": distributions,
            "affected_variables": affected_variables,
        }
        return df, params
    else:
        return df
