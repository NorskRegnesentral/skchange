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
def continuous_linear_trend_score(
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
    n_intervals = len(starts)
    n_columns = X.shape[1]
    scores = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, split, end = starts[i], splits[i], ends[i]
        start_values = X[start, :]
        split_values = X[split, :]
        first_interval_slopes = (split_values - start_values) / (split - start)
        second_interval_slopes = (X[end - 1, :] - split_values) / (end - 1 - split)
        first_interval_steps = np.arange(1, split - start)
        second_interval_steps = np.arange(1, end - 1 - split)
        # At the start, split, and end index, the fitted line is equal to the data.
        # Therefore, we only sum the squared differences for the remaining points.
        # The first sum goes over [start + 1, split), and the
        # second sum goes over [split + 1, end - 1).
        split_interval_cost = np.sum(
            np.square(
                X[(start + 1) : split, :]
                - (start_values + first_interval_steps[:, None] * first_interval_slopes)
            ),
            axis=0,
        ) + np.sum(
            np.square(
                X[(split + 1) : (end - 1), :]
                - (
                    split_values
                    + second_interval_steps[:, None] * second_interval_slopes
                )
            ),
            axis=0,
        )

        joint_interval_slopes = (X[end - 1, :] - start_values) / (end - 1 - start)
        joint_interval_steps = np.arange(1, end - start - 1)
        joint_interval_cost = np.sum(
            np.square(
                X[(start + 1) : (end - 1), :]
                - (start_values + joint_interval_steps[:, None] * joint_interval_slopes)
            ),
            axis=0,
        )

        scores[i, :] = joint_interval_cost - split_interval_cost

    return scores


class ContinuousLinearTrendScore(BaseChangeScore):
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
        return continuous_linear_trend_score(
            starts=starts, splits=splits, ends=ends, X=self._X
        )

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
