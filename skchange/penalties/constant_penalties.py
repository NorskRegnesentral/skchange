"""Constant penalties for change and anomaly detection."""

import numpy as np

from skchange.penalties.base import BasePenalty
from skchange.utils.validation.parameters import check_larger_than


class ConstantPenalty(BasePenalty):
    """Constant penalty."""

    penalty_type = "constant"

    def __init__(self, value: float, scale: float = 1.0):
        self.value = value
        super().__init__(scale)

        check_larger_than(self.value, 0.0, param_name="value")

    @property
    def base_values(self) -> np.ndarray:
        """Get the base penalty values.

        Returns
        -------
        base_values : np.ndarray
            1D array of base (unscaled) penalty values. The shape of the output depends
            on the `penalty_type`:

            * If ``"constant"``, the output is of shape ``(1,)``.
            * If ``"linear"`` or ``"nonlinear"``, the output is of shape ``(p,)``,
            where ``p`` is the number of variables/columns in the data being analysed.
        """
        return np.array([self.value])

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
            {"value": 10.0, "scale": 1.0},
            {"value": 1.0, "scale": 0.5},
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
    n : int
        Sample size.
    n_params : int
        Number of parameters per segment in the model across all variables.
    scale : float, optional, default=1.0
        Scaling factor for the penalty.
    """

    penalty_type = "constant"

    def __init__(self, n: int, n_params: int, scale: float = 1.0):
        self.n = n
        self.n_params = n_params
        super().__init__(scale)

        check_larger_than(self.n, 1, param_name="n")
        check_larger_than(self.n_params, 1, param_name="n_params")

    @property
    def base_values(self) -> np.ndarray:
        """Get the base penalty values.

        Returns
        -------
        base_values : np.ndarray
            Shape ``(1,)`` array with the base (unscaled) penalty values.
        """
        # +1 due to the additional change point parameter in the model.
        base_penalty = (self.n_params + 1) * np.log(self.n)
        return np.array([base_penalty])

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
            {"n": 100, "n_params": 1, "scale": 1.0},
            {"n": 1000, "n_params": 2, "scale": 0.5},
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
    n : int
        Sample size.
    n_params : int
        Number of parameters per segment in the model across all variables.
    scale : float, optional, default=1.0
        Scaling factor for the penalty.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       segment and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.
    """

    penalty_type = "constant"

    def __init__(self, n: int, n_params: int, scale: float = 1.0):
        self.n = n
        self.n_params = n_params
        super().__init__(scale)

        check_larger_than(self.n, 1, param_name="n")
        check_larger_than(self.n_params, 1, param_name="n_params")

    @property
    def base_values(self) -> np.ndarray:
        """Get the base penalty values.

        Returns
        -------
        base_values : np.ndarray
            Shape ``(1,)`` array with the base (unscaled) penalty values.
        """
        psi = np.log(self.n)
        base_penalty = self.n_params + 2 * np.sqrt(self.n_params * psi) + 2 * psi
        return np.array([base_penalty])

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
            {"n": 100, "n_params": 1, "scale": 1.0},
            {"n": 1000, "n_params": 2, "scale": 0.5},
        ]
        return params
