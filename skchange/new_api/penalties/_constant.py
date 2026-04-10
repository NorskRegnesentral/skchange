"""Constant penalties for change and anomaly detection."""

import numbers

import numpy as np

from skchange.new_api.utils._param_validation import Interval, validate_params


@validate_params(
    {
        "n_samples": [Interval(numbers.Integral, 1, None, closed="left")],
        "n_params": [Interval(numbers.Integral, 1, None, closed="left")],
        "additional_cpts": [Interval(numbers.Integral, 0, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def bic_penalty(n_samples: int, n_params: int = 1, additional_cpts: int = 1) -> float:
    """Create a Bayesian Information Criterion (BIC) penalty.

    The BIC penalty is a constant penalty given by
    ``(n_params + additional_cpts) * log(n_samples)``, where `n_samples` is the
    sample size, `n_params` is the number of parameters per segment in the model
    across all features, and `additional_cpts` is the number of additional change
    point parameters per segment. For change detection, this is 1.

    Parameters
    ----------
    n_samples : int
        Sample size.
    n_params : int, default=1
        Number of model parameters per segment.
    additional_cpts : int, default=1
        Number of additional change point parameters per segment. For change
        detection, this is 1.

    Returns
    -------
    float
        The BIC penalty value.

    Examples
    --------
    >>> bic_penalty(100)
    9.210340371976184
    """
    return (n_params + additional_cpts) * np.log(n_samples)


@validate_params(
    {
        "n_samples": [Interval(numbers.Integral, 1, None, closed="left")],
        "n_params": [Interval(numbers.Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def chi2_penalty(n_samples: int, n_params: int = 1) -> float:
    """Create a chi-square penalty.

    The penalty is the default penalty for the `CAPA` algorithm. It is described as
    "penalty regime 1" in the MVCAPA article [1]_. The penalty is based on a
    probability bound on the chi-squared distribution.

    The penalty is given by
    ``n_params + 2 * sqrt(n_params * log(n_samples)) + 2 * log(n_samples)``,
    where `n_samples` is the sample size and `n_params` is the total number of
    parameters per segment in the model across all features.

    Parameters
    ----------
    n_samples : int
        Sample size.
    n_params : int, default=1
        Number of model parameters per segment.

    Returns
    -------
    float
        The chi-square penalty value.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       segment and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.

    Examples
    --------
    >>> chi2_penalty(100, n_params=2)
    27.536...
    """
    psi = np.log(n_samples)
    return n_params + 2 * np.sqrt(n_params * psi) + 2 * psi
