"""Linear penalties for change and anomaly detection."""

import numbers

import numpy as np

from skchange.new_api.penalties._constant import chi2_penalty
from skchange.new_api.utils._param_validation import Interval, validate_params


@validate_params(
    {
        "n_features": [Interval(numbers.Integral, 1, None, closed="left")],
        "intercept": [Interval(numbers.Real, 0, None, closed="left")],
        "slope": [Interval(numbers.Real, 0, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def linear_penalty(n_features: int, intercept: float, slope: float) -> np.ndarray:
    """Create a linear penalty.

    The penalty is given by ``intercept + slope * (1, 2, ..., n_features)``, where
    `n_features` is the number of features/columns in the data being analysed. The
    penalty is non-decreasing.

    Parameters
    ----------
    n_features : int
        Number of features/columns in the data being analysed.
    intercept : float
        Intercept of the linear penalty.
    slope : float
        Slope of the linear penalty.

    Returns
    -------
    np.ndarray of shape (n_features,)
        The non-decreasing linear penalty values. Element ``i`` is the penalty for
        ``i+1`` features being affected by a change or anomaly.

    Examples
    --------
    >>> linear_penalty(3, 1.0, 2.0)
    array([3., 5., 7.])
    """
    return intercept + slope * np.arange(1, n_features + 1)


@validate_params(
    {
        "n_samples": [Interval(numbers.Integral, 1, None, closed="left")],
        "n_features": [Interval(numbers.Integral, 1, None, closed="left")],
        "n_params_per_feature": [Interval(numbers.Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def linear_chi2_penalty(
    n_samples: int, n_features: int, n_params_per_feature: int = 1
) -> np.ndarray:
    """Create a linear chi-square penalty.

    The penalty is a piece of the default penalty for the `MVCAPA` algorithm. It is
    described as "penalty regime 2" in the MVCAPA article [1]_, suitable for detecting
    sparse anomalies in the data. Sparse anomalies only affect a few features.

    Parameters
    ----------
    n_samples : int
        Sample size.
    n_features : int
        Number of features/columns in the data being analysed.
    n_params_per_feature : int, default=1
        Number of model parameters per feature and segment.

    Returns
    -------
    np.ndarray of shape (n_features,)
        The non-decreasing linear chi-square penalty values. Element ``i`` is the
        penalty for ``i+1`` features being affected by a change or anomaly.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       segment and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.

    Examples
    --------
    >>> linear_chi2_penalty(100, 3)
    array([...])
    """
    if n_features == 1:
        # Not valid for n_features == 1; fall back to constant chi2 penalty.
        return np.array([chi2_penalty(n_samples, n_params_per_feature)])

    psi = np.log(n_samples)
    component_penalty = 2 * np.log(n_params_per_feature * n_features)
    return 2 * psi + 2 * np.cumsum(np.full(n_features, component_penalty))
