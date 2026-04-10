"""Non-linear penalties for change and anomaly detection."""

import numbers

import numpy as np
from scipy.stats import chi2

from skchange.new_api.penalties._constant import chi2_penalty
from skchange.new_api.penalties._linear import linear_chi2_penalty
from skchange.new_api.utils._param_validation import Interval, validate_params


@validate_params(
    {
        "n_params_per_variable": [Interval(numbers.Integral, 1, None, closed="left")],
        "n": [Interval(numbers.Integral, 1, None, closed="left")],
        "p": [Interval(numbers.Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def nonlinear_chi2_penalty(n_params_per_variable: int, n: int, p: int) -> np.ndarray:
    """Create a nonlinear chi-square penalty.

    The penalty is a piece of the default penalty for the `MVCAPA` algorithm. It is
    described as "penalty regime 3" in the MVCAPA article [1]_, suitable for detecting
    both sparse and dense anomalies in the data.

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
        The non-decreasing nonlinear chi-square penalty values. Element ``i`` is the
        penalty for ``i+1`` variables being affected by a change or anomaly.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       segment and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.

    Examples
    --------
    >>> nonlinear_chi2_penalty(1, 100, 3)
    array([...])
    """
    if p == 1:
        # Not defined for p = 1; fall back to constant chi2 penalty.
        return np.array([chi2_penalty(n_params_per_variable, n)])

    def penalty_func(j: int) -> float:
        psi = np.log(n)
        c_j = chi2.ppf(1 - j / p, n_params_per_variable)
        f_j = chi2.pdf(c_j, n_params_per_variable)
        return (
            2 * (psi + np.log(p))
            + j * n_params_per_variable
            + 2 * p * c_j * f_j
            + 2
            * np.sqrt(
                (j * n_params_per_variable + 2 * p * c_j * f_j) * (psi + np.log(p))
            )
        )

    penalties = np.zeros(p, dtype=float)
    penalties[:-1] = np.vectorize(penalty_func)(np.arange(1, p))
    penalties[-1] = penalties[-2]
    return penalties


@validate_params(
    {
        "n_params_per_variable": [Interval(numbers.Integral, 1, None, closed="left")],
        "n": [Interval(numbers.Integral, 1, None, closed="left")],
        "p": [Interval(numbers.Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def mvcapa_penalty(n_params_per_variable: int, n: int, p: int) -> np.ndarray:
    """Create the default penalty for the MVCAPA algorithm.

    The penalty is the pointwise minimum of the constant, linear, and nonlinear
    chi-square penalties: `chi2_penalty`, `linear_chi2_penalty`, and
    `nonlinear_chi2_penalty`. It is the recommended penalty for the MVCAPA
    algorithm [1]_.

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
        The pointwise minimum penalty values. Element ``i`` is the penalty for
        ``i+1`` variables being affected by a change or anomaly.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       segment and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.

    Examples
    --------
    >>> mvcapa_penalty(1, 100, 3)
    array([...])
    """
    n_params_total = n_params_per_variable * p
    constant_part = chi2_penalty(n_params_total, n)
    linear_part = linear_chi2_penalty(n_params_per_variable, n, p)
    nonlinear_part = nonlinear_chi2_penalty(n_params_per_variable, n, p)
    return np.fmin(constant_part, np.fmin(linear_part, nonlinear_part))
