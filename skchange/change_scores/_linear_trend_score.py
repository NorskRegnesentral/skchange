"""Linear trend cost function.

This module contains the LinearTrendCost class, which is a cost function for
change point detection based on the squared error between data points
and a best fit linear trend line within each interval.
"""

import numpy as np

from ..utils.numba import njit
from ..utils.validation.enums import EvaluationType
from .base import BaseChangeScore


@njit
def linear_trend_score(
    starts: np.ndarray, splits: np.ndarray, ends: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Evaluate the continuous linear trend cost.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the first intervals (inclusive).
    splits : np.ndarray
        Split indices between the intervals (contained in second interval).
    ends : np.ndarray
        End indices of the second intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    ### NOTE: Assume 'time' is index of the data. i.e. time = 0, 1, 2, ..., len(X)-1
    ###       This assumption could be changed later.
    n_intervals = len(starts)
    n_columns = X.shape[1]
    scores = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, split, end = starts[i], splits[i], ends[i]
        split_interval_trend_data = np.zeros((end - start, 3))
        split_interval_trend_data[:, 0] = 1.0  # Intercept
        # Whole interval slope:
        split_interval_trend_data[:, 1] = np.arange(end - start)  # Time steps
        # Change in slope from the split point: trend data index starts at 0 from 'start'.
        split_interval_trend_data[(split - start) :, 2] = np.arange(end - split)

        # Calculate the slope and intercept for the whole interval:
        split_interval_linreg_res = np.linalg.lstsq(
            split_interval_trend_data, X[start:end, :]
        )
        split_interval_squared_residuals = split_interval_linreg_res[1]

        # By only regressing onto the first two columns, we can calculate the cost
        # without allowing for a change in slope at the split point.
        joint_interval_linreg_res = np.linalg.lstsq(
            split_interval_trend_data[:, [0, 1]], X[start:end, :]
        )
        joint_interval_squared_residuals = joint_interval_linreg_res[1]

        scores[i, :] = (
            joint_interval_squared_residuals - split_interval_squared_residuals
        )

    return scores


class BestFitLinearTrendScore(BaseChangeScore):
    """Continuous linear trend change score.

    This change score calculates the sum of squared errors between observed data
    and a continuous piecewise linear trend line connecting the `start`, `split`,
    and `end` points within each test interval, with a kink at the `split` point.

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
    ):
        super().__init__()

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

    def _evaluate(self, cuts: np.ndarray) -> np.ndarray:
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
        starts = cuts[:, 0]
        splits = cuts[:, 1]
        ends = cuts[:, 2]
        return linear_trend_score(starts=starts, splits=splits, ends=ends, X=self._X)

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
        params = [{}]
        return params
