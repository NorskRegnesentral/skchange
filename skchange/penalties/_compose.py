"""Composite penalties for change and anomaly detection."""

import numpy as np
import pandas as pd

from ..base import BaseIntervalScorer
from .base import BasePenalty


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

        if len(penalties) < 1:
            raise ValueError("penalties must contain at least one penalty")

        self._penalty_types = [penalty.penalty_type for penalty in self.penalties]
        if "nonlinear" in self._penalty_types:
            self.penalty_type = "nonlinear"
        elif "linear" in self._penalty_types:
            self.penalty_type = "linear"
        else:
            self.penalty_type = "constant"

    def _fit(
        self, X: pd.DataFrame | pd.Series | np.ndarray, scorer: BaseIntervalScorer
    ) -> "BasePenalty":
        """Fit the penalty to data and a scorer.

        This method should be implemented if more fitting is needed than just obtaining
        the number of samples and variables in the data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            The data to fit the penalty to.
        scorer : BaseIntervalScorer
            The interval scorer to fit the penalty to.

        Returns
        -------
        self
            Reference to self.
        """
        self.penalties[0].set_params(scale=1)
        self.penalties[0].fit(X, scorer)
        self._min_penalties = self.penalties[0].values
        if len(self.penalties) > 1:
            for penalty in self.penalties[1:]:
                penalty.set_params(scale=1)
                penalty.fit(X, scorer)
                self._min_penalties = np.minimum(self._min_penalties, penalty.values)
        return self

    @property
    def _base_values(self) -> np.ndarray:
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
            ChiSquarePenalty(),
            LinearChiSquarePenalty(),
            NonlinearChiSquarePenalty(),
        ]

        params = [
            {"penalties": penalties, "scale": 1.0},
            {"penalties": penalties, "scale": 2.0},
        ]
        return params
