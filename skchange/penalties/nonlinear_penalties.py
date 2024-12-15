"""Non-linear penalties for change and anomaly detection."""

import numpy as np
from scipy.stats import chi2

from skchange.penalties.base import BasePenalty
from skchange.utils.validation.parameters import check_larger_than, check_smaller_than


class IntermediateChiSquarePenalty(BasePenalty):
    """Sparse Chi-Square penalty.

    The penalty is a piece of the default penalty for the `MVCAPA` algorithm. It is
    described as "penalty regime 3" in the MVCAPA article [1]_. It is called
    "intermediate" because it is suitable for detecting both sparse and dense anomalies
    in the data. Sparse anomalies only affect a few variables, while dense anomalies
    affect all variables.

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
        check_smaller_than(self.n_params, 2, param_name="n_params")
        check_larger_than(self.p, 2, param_name="p")

        self._base_penalty_values = self._make_penalty(n, p, n_params)

    def _make_penalty(self, n: int, p: int, n_params: int) -> np.ndarray:
        def penalty_func(j: int) -> float:
            psi = np.log(n)
            c_j = chi2.ppf(1 - j / p, n_params)
            f_j = chi2.pdf(c_j, n_params)
            penalty = (
                2 * (psi + np.log(p))
                + j * n_params
                + 2 * p * c_j * f_j
                + 2 * np.sqrt((j * n_params + 2 * p * c_j * f_j) * (psi + np.log(p)))
            )
            return penalty

        penalties = np.zeros(self.p, dtype=float)
        penalties[:-1] = np.vectorize(penalty_func)(np.arange(1, self.p))
        # The penalty function is not defined for j = p, so we just duplicate the last
        # value.
        penalties[-1] = penalties[-2]
        return penalties

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
        return self._base_penalty_values
