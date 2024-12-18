"""Linear penalties for change and anomaly detection."""

import numpy as np

from skchange.penalties.base import BasePenalty
from skchange.utils.validation.parameters import check_larger_than


class LinearChiSquarePenalty(BasePenalty):
    """Linear Chi-square penalty.

    The penalty is a piece of the default penalty for the `MVCAPA` algorithm. It is
    described as "penalty regime 2" in the MVCAPA article [1]_, suitable for detecting
    sparse anomalies in the data. Sparse anomalies only affect a few variables.

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of variables/columns in the data.
    n_params : int, optional, default=1
        Number of parameters per variable and segment in the model.
    scale : float, optional, default=1.0
        Scaling factor for the penalty.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       segment and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.
    """

    penalty_type = "linear"

    def __init__(self, n: int, p: int, n_params: int = 1, scale: float = 1.0):
        self.n = n
        self.p = p
        self.n_params = n_params
        super().__init__(scale)

        check_larger_than(self.n, 1, param_name="n")
        check_larger_than(self.n_params, 1, param_name="n_params")
        check_larger_than(self.p, 1, param_name="p")

    @property
    def base_values(self) -> np.ndarray:
        """Get the base penalty values.

        Returns
        -------
        base_values : np.ndarray
            Shape ``(p,)`` array with the base (unscaled) penalty values, where ``p`` is
            the number of variables/columns in the data being analysed. Element ``i`` of
            the array is the base penalty value for ``i+1`` variables being affected by
            the change. The base penalty array is non-decreasing.
        """
        psi = np.log(self.n)
        component_penalty = 2 * np.log(self.n_params * self.p)
        base_penalties = 2 * psi + 2 * np.cumsum(np.full(self.p, component_penalty))
        return base_penalties

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
            {"n": 100, "p": 10, "n_params": 1, "scale": 1.0},
            {"n": 1000, "p": 3, "n_params": 2, "scale": 0.5},
        ]
        return params
