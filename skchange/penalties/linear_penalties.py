"""Linear penalties for change and anomaly detection."""

import numpy as np

from skchange.penalties.base import BasePenalty
from skchange.utils.validation.parameters import check_larger_than


class SparseChiSquarePenalty(BasePenalty):
    """Sparse Chi-Square penalty.

    The penalty is a piece of the default penalty for the `MVCAPA` algorithm. It is
    described as "penalty regime 2" in the MVCAPA article [1]_. It is called "sparse"
    because it is suitable for detecting sparse anomalies in the data, i.e., anomalies
    that affect only a few variables.

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
