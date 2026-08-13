"""Non-linear penalties for change and anomaly detection."""

import numbers

import numpy as np
from scipy.stats import chi2

from skchange.new_api.penalties._constant import chi2_penalty
from skchange.new_api.penalties._linear import linear_chi2_penalty
from skchange.new_api.utils._param_validation import Interval, validate_params


@validate_params(
    {
        "n_samples": [Interval(numbers.Integral, 1, None, closed="left")],
        "n_features": [Interval(numbers.Integral, 1, None, closed="left")],
        "n_params_per_feature": [Interval(numbers.Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def nonlinear_chi2_penalty(
    n_samples: int, n_features: int, n_params_per_feature: int = 1
) -> np.ndarray:
    """Create a nonlinear chi-square penalty.

    The penalty is a piece of the default penalty for the `MVCAPA` algorithm. It is
    described as "penalty regime 3" in the MVCAPA article [1]_, suitable for detecting
    both sparse and dense anomalies in the data.

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
        The non-decreasing nonlinear chi-square penalty values. Element ``i`` is the
        penalty for ``i+1`` features being affected by a change or anomaly.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       segment and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.

    Examples
    --------
    >>> nonlinear_chi2_penalty(100, 3)
    array([...])
    """
    if n_features == 1:
        # Not defined for n_features = 1; fall back to constant chi2 penalty.
        return np.array([chi2_penalty(n_samples, n_params_per_feature)])

    def penalty_func(j: int) -> float:
        psi = np.log(n_samples)
        c_j = chi2.ppf(1 - j / n_features, n_params_per_feature)
        f_j = chi2.pdf(c_j, n_params_per_feature)
        return (
            2 * (psi + np.log(n_features))
            + j * n_params_per_feature
            + 2 * n_features * c_j * f_j
            + 2
            * np.sqrt(
                (j * n_params_per_feature + 2 * n_features * c_j * f_j)
                * (psi + np.log(n_features))
            )
        )

    penalties = np.zeros(n_features, dtype=float)
    penalties[:-1] = np.vectorize(penalty_func)(np.arange(1, n_features))
    penalties[-1] = penalties[-2]
    return penalties


@validate_params(
    {
        "n_samples": [Interval(numbers.Integral, 1, None, closed="left")],
        "n_features": [Interval(numbers.Integral, 1, None, closed="left")],
        "n_params_per_feature": [Interval(numbers.Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def mvcapa_penalty(
    n_samples: int, n_features: int, n_params_per_feature: int = 1
) -> np.ndarray:
    """Create the default penalty for the MVCAPA algorithm.

    The penalty is the pointwise minimum of the constant, linear, and nonlinear
    chi-square penalties: `chi2_penalty`, `linear_chi2_penalty`, and
    `nonlinear_chi2_penalty`. It is the recommended penalty for the MVCAPA
    algorithm [1]_.

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
        The pointwise minimum penalty values. Element ``i`` is the penalty for
        ``i+1`` features being affected by a change or anomaly.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       segment and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.

    Examples
    --------
    >>> mvcapa_penalty(100, 3)
    array([...])
    """
    n_params_total = n_params_per_feature * n_features
    constant_part = chi2_penalty(n_samples, n_params_total)
    linear_part = linear_chi2_penalty(n_samples, n_features, n_params_per_feature)
    nonlinear_part = nonlinear_chi2_penalty(n_samples, n_features, n_params_per_feature)
    return np.fmin(constant_part, np.fmin(linear_part, nonlinear_part))
