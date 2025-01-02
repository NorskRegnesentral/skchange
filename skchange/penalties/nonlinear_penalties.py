"""Non-linear penalties for change and anomaly detection."""

from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import chi2

from skchange.base import BaseIntervalScorer
from skchange.penalties.base import BasePenalty


def check_penalty_array(penalty_array: np.ndarray) -> None:
    """Check the base penalty values.

    Parameters
    ----------
    penalty_array : np.ndarray
        Shape ``(p,)`` array with penalty values, where ``p`` is
        the number of variables/columns in the data being analysed. Element ``i`` of
        the array is the base penalty value for ``i+1`` variables being affected by
        the change. The penalty array is non-decreasing.
    """
    if not isinstance(penalty_array, np.ndarray):
        raise TypeError("penalty_array must be a numpy array")
    if penalty_array.ndim != 1:
        raise ValueError("penalty_array must be a 1D array")
    if penalty_array.size < 1:
        raise ValueError("penalty_array must have at least one element")
    if not np.all(penalty_array >= 0.0):
        raise ValueError("penalty_array must be non-negative")
    if not np.all(np.diff(penalty_array) >= 0):
        raise ValueError("penalty_array must be non-decreasing")


class NonlinearPenalty(BasePenalty):
    """Non-linear penalty."""

    penalty_type = "nonlinear"

    def __init__(self, base_values: np.ndarray, scale: float = 1.0):
        self.base_values = base_values
        super().__init__(scale)

        check_penalty_array(self.base_values)

    def _fit(
        self, X: Union[pd.DataFrame, pd.Series, np.ndarray], scorer: BaseIntervalScorer
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
        p = X.shape[1]
        if p == 1:
            self.base_values_ = np.array([self.base_values[-1]])
        elif self.base_values.size != p:
            self.base_values_ = np.interp(
                np.linspace(0, self.base_values.size - 1, p),
                np.arange(self.base_values.size),
                self.base_values,
            )
        else:
            self.base_values_ = self.base_values

        return self

    @property
    def _base_values(self) -> np.ndarray:
        """Get the base penalty values.

        Returns
        -------
        base_values : np.ndarray
            Shape ``(p,)`` array with the base (unscaled) penalty values, where ``p`` is
            the number of variables/columns in the data being analysed. Element ``i`` of
            the array is the base penalty value for ``i+1`` variables being affected by
            the change. The base penalty array is non-decreasing.
        """
        return self.base_values_

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
            {"base_values": np.array([1.0, 2.0, 3.0, 5.0, 5.0]), "scale": 1.0},
            {"base_values": np.array([0.5, 1.0]), "scale": 0.5},
        ]
        return params


class NonlinearChiSquarePenalty(BasePenalty):
    """Nonlinear Chi-square penalty.

    The penalty is a piece of the default penalty for the `MVCAPA` algorithm. It is
    described as "penalty regime 3" in the MVCAPA article [1]_, suitable for detecting
    both sparse and dense anomalies in the data. Sparse anomalies only affect a few
    variables, while dense anomalies affect many/all variables.

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

    penalty_type = "nonlinear"

    def __init__(self, scale: float = 1.0):
        super().__init__(scale)

    def _fit(
        self, X: Union[pd.DataFrame, pd.Series, np.ndarray], scorer: BaseIntervalScorer
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
        if self.n_params_per_variable > 2:
            raise ValueError(
                "NonlinearChiSquarePenalty can only be used with scorers that have at"
                " most 2 parameters per variable (scorer.get_param_size(1) <= 2)"
            )

        self._base_penalty_values = self._make_penalty(
            self.n, self.p, self.n_params_per_variable
        )
        return self

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
    def _base_values(self) -> np.ndarray:
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
