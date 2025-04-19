"""Constant penalties for change and anomaly detection."""

import numpy as np

from ..utils.validation.parameters import check_larger_than
from .base import BasePenalty


def make_bic_penalty(n_params: int, n: int):
    """Create a BIC penalty.

    Parameters
    ----------
    n_params : int
        Total number of parameters of the model in each segment.
    n : int
        Sample size.

    Returns
    -------
    penalty : float
        The BIC penalty value.
    """
    return (n_params + 1) * np.log(n)  # +1 due to the change point parameter


class ConstantPenalty(BasePenalty):
    """Constant penalty."""

    penalty_type = "constant"

    def __init__(self, base_value: float, scale: float = 1.0):
        self.base_value = base_value
        super().__init__(scale)

        check_larger_than(0.0, self.base_value, "base_value")

    @property
    def _base_values(self) -> np.ndarray:
        """Get the base penalty values.

        Returns
        -------
        base_values : np.ndarray
            Shape ``(1,)`` array with the base (unscaled) penalty values.
        """
        return self.base_value

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for penalties.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            {"base_value": 10.0, "scale": 1.0},
            {"base_value": 1.0, "scale": 0.5},
        ]
        return params


class BICPenalty(BasePenalty):
    """Bayesian Information Criterion (BIC) penalty.

    The BIC penalty is a constant penalty given by ``(n_params + 1) * log(n)``, where
    `n` is the sample size and `n_params` is the number of parameters per segment in the
    model across all variables. The ``+ 1`` term accounts for the change point
    parameter.

    Parameters
    ----------
    scale : float, optional, default=1.0
        Scaling factor for the penalty.
    """

    penalty_type = "constant"

    def __init__(self, scale: float = 1.0):
        super().__init__(scale)

    @property
    def _base_values(self) -> np.ndarray:
        """Get the base penalty values.

        Returns
        -------
        base_values : np.ndarray
            Shape ``(1,)`` array with the base (unscaled) penalty values.
        """
        # +1 due to the additional change point parameter in the model.
        base_penalty = (self.n_params_total_ + 1) * np.log(self.n_)
        return base_penalty

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for penalties.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            {"scale": 1.0},
            {"scale": 0.5},
        ]
        return params


class ChiSquarePenalty(BasePenalty):
    """Penalty based on a probability bound on the chi-squared distribution.

    The penalty is the default penalty for the `CAPA` algorithm. It is described as
    "penalty regime 1" in the MVCAPA article [1]_.

    The penalty is given by ``n_params + 2 * sqrt(n_params * log(n)) + 2 * log(n)``,
    where `n` is the sample size and `n_params` is the total number of parameters per
    segment in the model across all variables.

    Parameters
    ----------
    scale : float, optional, default=1.0
        Scaling factor for the penalty.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       segment and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.
    """

    penalty_type = "constant"

    def __init__(self, scale: float = 1.0):
        super().__init__(scale)

    @property
    def _base_values(self) -> np.ndarray:
        """Get the base penalty values.

        Returns
        -------
        base_values : np.ndarray
            Shape ``(1,)`` array with the base (unscaled) penalty values.
        """
        psi = np.log(self.n_)
        base_penalty = (
            self.n_params_total_ + 2 * np.sqrt(self.n_params_total_ * psi) + 2 * psi
        )
        return base_penalty

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for penalties.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            {"scale": 1.0},
            {"scale": 0.5},
        ]
        return params
