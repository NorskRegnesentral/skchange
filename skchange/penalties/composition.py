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

        self._penalty_types = [penalty.penalty_type for penalty in self.penalties]
        if "nonlinear" in self._penalty_types:
            self.penalty_type = "nonlinear"
        elif "linear" in self._penalty_types:
            self.penalty_type = "linear"
        else:
            self.penalty_type = "constant"

        ps = [getattr(penalty, "p", 1) for penalty in self.penalties]
        unique_ps = np.unique(ps)
        if unique_ps.size > 2 or (unique_ps.size == 2 and 1 not in unique_ps):
            raise ValueError(
                "All non-constant penalties must be configured for the same number of"
                " variables `p`"
            )
        self.p = max(ps)

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
        from skchange.penalties import (
            ChiSquarePenalty,
            LinearChiSquarePenalty,
            NonlinearChiSquarePenalty,
        )

        penalties = [
            ChiSquarePenalty(100, 10, 1),
            LinearChiSquarePenalty(100, 10, 1),
            NonlinearChiSquarePenalty(100, 10, 1),
        ]

        params = [
            {"penalties": penalties, "scale": 1.0},
            {"penalties": penalties, "scale": 2.0},
        ]
        return params
