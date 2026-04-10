"""Linear penalties for change and anomaly detection."""

import numbers

import numpy as np

from skchange.new_api.utils._param_validation import Interval, validate_params


@validate_params(
    {
        "intercept": [Interval(numbers.Real, 0, None, closed="left")],
        "slope": [Interval(numbers.Real, 0, None, closed="left")],
        "p": [Interval(numbers.Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def linear_penalty(intercept: float, slope: float, p: int) -> np.ndarray:
    """Create a linear penalty.

    The penalty is given by ``intercept + slope * (1, 2, ..., p)``, where `p` is the
    number of variables/columns in the data being analysed. The penalty is
    non-decreasing.

    Parameters
    ----------
    intercept : float
        Intercept of the linear penalty.
    slope : float
        Slope of the linear penalty.
    p : int
        Number of variables/columns in the data being analysed.

    Returns
    -------
    np.ndarray of shape (p,)
        The non-decreasing linear penalty values. Element ``i`` is the penalty for
        ``i+1`` variables being affected by a change or anomaly.

    Examples
    --------
    >>> linear_penalty(1.0, 2.0, 3)
    array([3., 5., 7.])
    """
    return intercept + slope * np.arange(1, p + 1)


@validate_params(
    {
        "n_params_per_variable": [Interval(numbers.Integral, 1, None, closed="left")],
        "n": [Interval(numbers.Integral, 1, None, closed="left")],
        "p": [Interval(numbers.Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def linear_chi2_penalty(n_params_per_variable: int, n: int, p: int) -> np.ndarray:
    """Create a linear chi-square penalty.

    The penalty is a piece of the default penalty for the `MVCAPA` algorithm. It is
    described as "penalty regime 2" in the MVCAPA article [1]_, suitable for detecting
    sparse anomalies in the data. Sparse anomalies only affect a few variables.

    Parameters
    ----------
    n_params_per_variable : int
        Number of model parameters per variable and segment.
    n : int
        Sample size.
    p : int
        Number of variables/columns in the data being analysed.

    Returns
    -------
    np.ndarray of shape (p,)
        The non-decreasing linear chi-square penalty values. Element ``i`` is the
        penalty for ``i+1`` variables being affected by a change or anomaly.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       segment and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.

    Examples
    --------
    >>> linear_chi2_penalty(1, 100, 3)
    array([...])
    """
    psi = np.log(n)
    component_penalty = 2 * np.log(n_params_per_variable * p)
    return 2 * psi + 2 * np.cumsum(np.full(p, component_penalty))
