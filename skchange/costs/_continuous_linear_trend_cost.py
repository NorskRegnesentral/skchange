"""Linear trend cost function.

This module contains the LinearTrendCost class, which is a cost function for
change point detection based on the squared error between data points
and a best fit linear trend line within each interval.
"""

import numpy as np

from ..utils.numba import njit
from ..utils.validation.enums import EvaluationType
from .base import BaseCost


@njit
def continuous_linear_trend_cost_mle(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Evaluate the continuous linear trend cost.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        intercepts = X[start, :]
        slopes = (X[end - 1, :] - intercepts) / (end - start - 1)
        steps = np.arange(end - start)
        costs[i, :] = np.sum(
            np.square(X[start:end, :] - (intercepts + steps[:, None] * slopes)),
            axis=0,
        )

    return costs


class ContinuousLinearTrendCost(BaseCost):
    """Continuous linear trend squared residual cost.

    This cost function calculates the sum of squared errors between data points
    and a linear trend line connecting the first and last points of each interval.
    For each interval, a straight line is drawn from the first to the last point
    of each signal, and the squared differences between the actual data points
    and this line are summed. This construction ensures that the piecewise linear
    trend is continuous when considering the segmentation of each signal
    as a whole, while allowing the slopes to change at each change point.

    # REF: https://www.tandfonline.com/doi/pdf/10.1080/10618600.2018.1512868
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
    }

    evaluation_type = EvaluationType.UNIVARIATE
    supports_fixed_params = False

    def __init__(
        self,
        param=None,
    ):
        super().__init__(param)

    def _fit(self, X: np.ndarray, y=None):
        """Fit the cost.

        This method stores the input data for later cost evaluation.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y: None
            Ignored. Included for API consistency by convention.
        """
        return self

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the continuous linear trend parameters.

        Evaluates the cost for `X[start:end]` for each each start, end in starts, ends.
        On each interval, a trend line connecting the first and last points is created,
        and the squared differences between the data points and the
        trend line are summed. This is done for each column in the data.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of
            columns is equal to the number of columns in the input data.
        """
        return continuous_linear_trend_cost_mle(starts, ends, self._X)

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate.

        The size of each interval is defined as ``cuts[i, 1] - cuts[i, 0]``.
        For continuous linear trend, we need at least 2 points.

        Returns
        -------
        int
            The minimum valid size of an interval to evaluate.
        """
        return 2  # Need at least 2 points to define a line

    def get_param_size(self, p: int) -> int:
        """Get the number of parameters in the cost function.

        Parameters
        ----------
        p : int
            Number of variables in the data.

        Returns
        -------
        int
            Number of parameters in the cost function.
        """
        # In each interval we need 2 parameters per column: slope and intercept.
        # TODO: Consider impact of continuity constraint on number of parameters?
        return 2 * p

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class
        """
        params = [
            {"param": None},
            # Conformance tests want at least two parameter sets.
            {"param": None},
        ]
        return params
