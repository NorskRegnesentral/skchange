"""Composite penalties for change and anomaly detection."""

import numpy as np

from skchange.penalties.base import BasePenalty


class MinimumPenalty(BasePenalty):
    """Pointwise minimum of two penalties.

    This penalty is the pointwise minimum of several. It is useful when combining two
    or more penalties with different properties, e.g., a penalty that is good at
    detecting sparse anomalies and a penalty that is good at detecting dense anomalies.

    Parameters
    ----------
    penalties : list[BasePenalty]
        List of penalties to combine.
    scale : float, optional, default=1.0
        Scaling factor for the penalty.
    """

    penalty_type = None

    def __init__(self, penalties: list[BasePenalty], scale: float = 1.0):
        super().__init__(scale)
        self.penalties = penalties

        if len(penalties) < 2:
            raise ValueError("penalties must contain at least two penalties")

        self._penalty_types = {penalty.penalty_type for penalty in self.penalties}
        if "nonlinear" in self._penalty_types:
            self.penalty_type = "nonlinear"
        elif "linear" in self._penalty_types:
            self.penalty_type = "linear"
        else:
            self.penalty_type = "constant"

        # Compute the pointwise minimum of the base penalty values here to avoid
        # recomputing it at each call to `values`.
        self._min_penalties = self.penalties[0].base_values
        for penalty in self.penalties[1:]:
            self._min_penalties = np.minimum(self._min_penalties, penalty.base_values)

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
        return self._min_penalties
