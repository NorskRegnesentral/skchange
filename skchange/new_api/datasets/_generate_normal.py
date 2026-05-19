"""Data generators for piecewise normal data."""

__author__ = ["Tveten"]

from numbers import Number

import numpy as np
import scipy.stats

from ._generate import generate_piecewise_data
from ._utils import check_random_generator, check_segment_lengths, recycle_list


def get_n_variables(
    n_variables: int,
    means: float | np.ndarray | list[float] | list[np.ndarray] | None = None,
    variances: float | np.ndarray | list[float] | list[np.ndarray] | None = None,
) -> int:
    """Derive the number of variables from the input parameters."""
    # Convert to list if not, to make the rest of the code easier.
    if not isinstance(means, list):
        means = [means]
    if not isinstance(variances, list):
        variances = [variances]

    mean_n_vars = max(
        len(mean) if isinstance(mean, (list, np.ndarray)) else 1 for mean in means
    )
    var_n_vars = max(
        len(var) if isinstance(var, (list, np.ndarray)) else 1 for var in variances
    )
    return int(max(n_variables, mean_n_vars, var_n_vars))


def _check_affected_variables(
    proportion_affected: float | list[float] | np.ndarray | None,
    randomise_affected_variables: bool,
    n_segments: int,
    n_variables: int,
    random_state: np.random.Generator,
) -> list[np.ndarray]:
    """Check and return affected variables for each segment."""
    if proportion_affected is None:
        proportion_affected = scipy.stats.uniform(1e-8, 1).rvs(
            size=n_segments, random_state=random_state
        )
    elif isinstance(proportion_affected, Number):
        proportion_affected = [proportion_affected]

    if len(proportion_affected) == 0:
        raise ValueError("proportion_affected cannot be an empty list or np.array.")

    proportion_affected = recycle_list(list(proportion_affected), n_segments)

    for prop in proportion_affected:
        if not (0 < prop <= 1):
            raise ValueError(
                "Proportion of affected variables must be between (0, 1]."
                f" Got `proportion_affected`={prop}."
            )

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


def _check_means(
    means: float | np.ndarray | list[float] | list[np.ndarray] | None,
    n_segments: int,
    n_variables: int,
    affected_variables: list[np.ndarray],
    random_state: int | np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Check and return means for each segment."""
    if means is None or isinstance(means, (Number, np.ndarray)):
        means = [means] * n_segments

    n_variables = int(n_variables)
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

        if _mean.shape[0] != int(n_variables):
            raise ValueError(
                "Mean vector must have the same length as the number of variables."
                f" Got mean={_mean} with shape {_mean.shape}"
                f" and n_variables={n_variables}."
            )
        _means.append(_mean)

    _means = _means[1:]  # The first element is just to initialize the loop.

    # Means are recycled to match the number of segments. Will only have an effect in
    # cases where `means` is a list and the list is shorter than `n_segments`.
    _means = recycle_list(_means, n_segments)
    return _means


def _check_variances(
    variances: float | np.ndarray | list[float] | list[np.ndarray] | None,
    n_segments: int,
    n_variables: int,
    affected_variables: list[np.ndarray],
    random_state: int | np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Create covariance matrices for each segment."""
    if variances is None or isinstance(variances, (Number, np.ndarray)):
        variances = [variances] * n_segments

    n_variables = int(n_variables)
    _vars = [np.ones(n_variables)]  # Initialize for the loop to work.
    _covs = []
    for cov, affected in zip(variances, affected_variables):
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
                f" Got covariance matrix with shape {_var.shape} and p={n_variables}."
            )

        if not np.allclose(_cov, _cov.T, atol=1e-8):
            raise ValueError("Covariance matrix must be symmetric.")

        eigvals = np.linalg.eigvalsh(_cov)
        if np.any(eigvals <= 0):
            raise ValueError("Covariance matrix must be positive definite.")

        _covs.append(_cov)

    # Covariance matrices are recycled to match the number of segments. Will only have
    # an effect in cases where `variances` is a list and the list is shorter than
    # `n_segments`.
    _covs = recycle_list(_covs, n_segments)
    return _covs


def generate_piecewise_normal_data(
    means: float | np.ndarray | list[float] | list[np.ndarray] | None = None,
    variances: float | np.ndarray | list[float] | list[np.ndarray] | None = 1.0,
    lengths: int | list[int] | np.ndarray | None = None,
    *,
    n_segments: int = 3,
    n_samples: int = 100,
    n_variables: int = 1,
    proportion_affected: float | list[float] | np.ndarray | None = None,
    randomise_affected_variables: bool = False,
    seed: int | np.random.Generator | None = None,
    return_params: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Generate piecewise multivariate normal data.

    Generates piecewise multivariate normal data, where the distributional changes
    from one segment to another may be sparse. E.g., the difference between two
    mean vectors may only have a few non-zero elements.

    Parameters
    ----------
    means : float, list of float, or list of np.ndarray, optional (default=None)
        Means for each segment.
        They are recycled to match the number of segments specified by `lengths` or
        `n_segments`.
        If floats, they are used for all affected variables (see `proportion_affected`)
        If None, random means are generated from a normal distribution with mean 0
        and standard deviation 2.

    variances : float, list of float, or list of np.ndarray, optional (default=1.0)
        Variances or covariance matrices for each segment. Vectors are treated as
        diagonal covariance matrices.
        They are recycled to match the number of segments specified by `lengths` or
        `n_segments`.
        If floats, they are used for all affected variables (see `proportion_affected`)
        If None, random variances are generated from a chi-squared distribution with
        2 degrees of freedom.

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

    n_variables : int, optional (default=1)
        Number of variables (columns) in the generated data.

    proportion_affected: float, list of float, or np.ndarray, optional (default=None)
        Proportion of variables affected by each change.
        I.e., the proportion of non-zero elements in the differences between adjacent
        means or variances.
        Only applies when `means` and `variances` are None or floats and
        `n_variables > 1`.
        All proportions must be in (0, 1].
        The number of affected variables is determined as
        `int(np.ceil(n_variables * proportion_affected))`.
        The proportions are recycled to match the number of segments specified by
        `lengths` or `n_segments`.
        If None, a random proportion of variables is affected.

    randomise_affected_variables : bool, optional (default=False)
        If True, the affected variables are randomly selected for each change point.
        If False, the first variables are affected.

    seed : np.random.Generator | int | None, optional
        Seed for the random number generator or a numpy random generator instance.
        If specified, this ensures reproducible output across multiple calls.

    return_params : bool, optional (default=False)
        If True, the function returns a tuple of the generated DataFrame and a
        dictionary with the parameters used to generate the data.

    Returns
    -------
    np.ndarray of shape (n_samples, n_variables)
        Array with generated data.

    dict
        Dictionary containing the parameters used to generate the data.
        Keys: `"n_segments"`, `"n_samples"`, `"means"`, `"variances"`,
        `"lengths"`, `"change_points"` (the start indices of each segment), and
        `"affected_variables"` (which variables among 0:n_variables are affected by
        each change).
        Returned only if `return_params` is True.

    Examples
    --------
    >>> from skchange.new_api.datasets import generate_piecewise_normal_data
    >>> X = generate_piecewise_normal_data(
    ...     means=[0, 5], lengths=5, n_segments=2, n_variables=1, seed=0
    ... )
    >>> X.shape
    (10, 1)
    """
    random_generator = check_random_generator(seed)
    if n_variables < 1:
        raise ValueError("n_variables must be at least 1.")

    lengths = check_segment_lengths(
        lengths, n_segments, n_samples, seed=random_generator
    )
    n_segments = len(lengths)

    _n_variables = get_n_variables(n_variables, means, variances)
    affected_variables = _check_affected_variables(
        proportion_affected,
        randomise_affected_variables,
        n_segments,
        _n_variables,
        random_generator,
    )
    means = _check_means(
        means,
        n_segments,
        _n_variables,
        affected_variables,
        random_generator,
    )
    covs = _check_variances(
        variances,
        n_segments,
        _n_variables,
        affected_variables,
        random_generator,
    )
    distributions = [
        scipy.stats.multivariate_normal(mean=mean, cov=cov)
        for mean, cov in zip(means, covs)
    ]
    X, _params = generate_piecewise_data(
        distributions=distributions,
        lengths=lengths,
        seed=random_generator,
        return_params=True,
    )
    if return_params:
        params = {
            "n_segments": n_segments,
            "n_samples": _params["n_samples"],
            "means": means,
            "variances": covs,
            "lengths": _params["lengths"],
            "change_points": _params["change_points"],
            "affected_variables": affected_variables,
        }
        return X, params
    else:
        return X
