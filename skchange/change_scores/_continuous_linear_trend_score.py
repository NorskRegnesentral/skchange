"""Linear trend cost function.

This module contains the LinearTrendCost class, which is a cost function for
change point detection based on the squared error between data points
and a best fit linear trend line within each interval.
"""

import numpy as np

from ..utils.numba import njit
from ..utils.validation.enums import EvaluationType
from ..utils.validation.parameters import check_data_column
from .base import BaseChangeScore


@njit
def lin_reg_cont_piecewise_linear_trend_score(
    starts: np.ndarray,
    splits: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    times: np.ndarray,
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
    times : np.ndarray, optional
        Time steps corresponding to the data points. If the data points
        are evenly spaced, instead call the optimized NOT-score function.

    Returns
    -------
    scores : np.ndarray
        A 2D array of scores. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    scores = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, split, end = starts[i], splits[i], ends[i]
        split_interval_trend_data = np.zeros((end - start, 3))
        split_interval_trend_data[:, 0] = 1.0  # Intercept

        # Whole interval slope:
        # split_interval_trend_data[:, 1] = np.arange(end - start)  # Time steps
        split_interval_trend_data[:, 1] = times[start:end]  # Time steps

        # Change in slope from the 'split' index:
        # Continuous at the first point of the second interal, [split, end - 1]:
        # trend data index starts at 0 from 'start'.
        # split_interval_trend_data[(split - start) :, 2] = np.arange(end - split)
        split_interval_trend_data[(split - start) :, 2] = (
            times[split:end] - times[split]
        )

        ### THIS IS WHAT the 'NOT' people DO: ###
        # Change in slope from 'split - 1' index:
        # Continuous in the last point of the first interval, [start, split - 1]:
        # trend data index starts at 0 from 'start'.
        # split_interval_trend_data[(split-start):, 2] = np.arange(1, end-split+1)
        # split_interval_trend_data[(split - start) :, 2] = (
        #     times[split:end] - times[split - 1]
        # )

        # Calculate the slope and intercept for the whole interval:
        split_interval_linreg_res = np.linalg.lstsq(
            split_interval_trend_data, X[start:end, :]
        )
        split_interval_squared_residuals = split_interval_linreg_res[1]

        # By only regressing onto the first two columns, we can calculate the cost
        # without allowing for a change in slope at the split point.
        joint_interval_linreg_res = np.linalg.lstsq(
            split_interval_trend_data[:, np.array([0, 1])], X[start:end, :]
        )
        joint_interval_squared_residuals = joint_interval_linreg_res[1]

        scores[i, :] = (
            joint_interval_squared_residuals - split_interval_squared_residuals
        )

    return scores


@njit
def continuous_piecewise_linear_trend_squared_contrast(
    signal: np.ndarray,
    first_interval_inclusive_start: int,
    second_interval_inclusive_start: int,
    non_inclusive_end: int,
):
    # Assume 'start' is the first index of the data, perform inner product with the
    # desired segment of the data to get the cost.
    assert (
        first_interval_inclusive_start + 1
        < second_interval_inclusive_start
        < non_inclusive_end
    )
    ## Translate named parameters to the NOT-paper sytax.
    ## We are zero-indexing the data, whilst the paper is one-indexing.
    s = first_interval_inclusive_start - 1
    # Add one to NOT split index to account for their different definition
    # of where the change in slope starts from.
    b = second_interval_inclusive_start - 1 + 1
    e = non_inclusive_end - 1
    l = e - s
    alpha = np.sqrt(
        6.0 / (l * (l**2 - 1) * (1 + (e - b + 1) * (b - s) + (e - b) * (b - s - 1)))
    )
    beta = np.sqrt(((e - b + 1.0) * (e - b)) / ((b - s - 1.0) * (b - s)))

    first_interval_slope = 3.0 * (b - s) + (e - b) - 1.0
    first_interval_constant = b * (e - s - 1.0) + 2.0 * (s + 1.0) * (b - s)

    second_interval_slope = 3.0 * (e - b) + (b - s) + 1.0
    second_interval_constant = b * (e - s - 1.0) + 2.0 * e * (e - b + 1)

    # Accumulate the contrast value inner product:
    contrast = 0.0
    for t in range(s + 1, b + 1):
        contrast += (
            alpha * beta * (first_interval_slope * t - first_interval_constant)
        ) * signal[t - first_interval_inclusive_start]

    for t in range(b + 1, e + 1):
        contrast += (
            (-alpha / beta) * (second_interval_slope * t - second_interval_constant)
        ) * signal[t - first_interval_inclusive_start]

    return np.square(contrast)


@njit
def analytical_cont_piecewise_linear_trend_score(
    starts: np.ndarray, splits: np.ndarray, ends: np.ndarray, X: np.ndarray
):
    """Evaluate the continuous piecewise linear trend cost.

    Using the analytical solution, this function evaluates the cost for
    `X[start:end]` for each each `[start, split, end]` triplett in `cuts`.

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
    scores : np.ndarray
        A 2D array of scores. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    scores = np.zeros((len(starts), X.shape[1]))
    for i in range(len(starts)):
        start, split, end = starts[i], splits[i], ends[i]
        for j in range(X.shape[1]):
            scores[i, j] = continuous_piecewise_linear_trend_squared_contrast(
                X[start:end, j],
                first_interval_inclusive_start=start,
                second_interval_inclusive_start=split,
                non_inclusive_end=end,
            )

    return scores


class ContinuousLinearTrendScore(BaseChangeScore):
    """Continuous linear trend change score.

    This change score calculates the difference in the squared error between
    observed data and a two parameter linear trend accross the whole interval,
    with the squared error between a three parameter linear trend with an added
    kink at the split point. The cost is calculated for each column in the data.

    By default time steps are assumed to be evenly spaced. If a time column is
    provided, its time steps are used to calculate the linear trends.

    When a time columns is not provided, an analytical solution is used to calculate
    the score for each column in the data, courtesy of [1]_. Otherwise, two linear
    regression problems are solved for each interval, one with a kink at the split
    point and one without.

    Parameters
    ----------
    time_column : str, optional
            Name of the time column in the data. If provided, the time steps are used to
            calculate the linear trends. If not provided, the time steps are assumed to
            be evenly spaced.

    References
    ----------
    .. [1] Baranowski, R., Chen, Y., & Fryzlewicz, P. (2019). Narrowest-over-threshold \
    detection of multiple change points and change-point-like features. Journal of the \
    Royal Statistical Society Series B: Statistical Methodology, 81(3), 649-672.
    ----------

    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
    }

    evaluation_type = EvaluationType.UNIVARIATE
    supports_fixed_params = False

    def __init__(
        self,
        time_column: str | None = None,
    ):
        super().__init__()
        self.time_column = time_column
        self.time_column_idx = None
        self._time_stamps = None

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
        if self.time_column is not None:
            self.time_column_idx = check_data_column(
                self.time_column, "Time", X, self._X_columns
            )
        else:
            self.time_column_idx = None

        if self.time_column_idx is not None:
            self._time_stamps = X[:, self.time_column_idx]
        else:
            # No provided time column or fixed parameters, so we assume
            # the time steps are [0, 1, 2, ..., n-1] for each segment.
            self._time_stamps = None

        if self.time_column_idx is not None:
            piecewise_linear_trend_columns = np.delete(
                np.arange(X.shape[1]), self.time_column_idx
            )
            self._piecewise_linear_trend_data = X[:, piecewise_linear_trend_columns]
        else:
            self._piecewise_linear_trend_data = X
        return self

    def _evaluate(self, cuts: np.ndarray) -> np.ndarray:
        """Evaluate the continuous piecewise linear trend scores.

        Evaluates the score on `X[start:end]` for each each `[start, split, end]`
        triplett in cuts.  On each interval, the difference in summed squared
        residuals between the best fit linear trend accross the whole interval
        and the best fit linear trend with a kink at the split point
        is calculated. The score is calculated for each column in the data.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array with three columns of integer locations.
            The first column is the ``start``, the second is the ``split``, and the
            third is the ``end`` of the interval to evaluate the score on.

        Returns
        -------
        scores : np.ndarray
            A 2D array of costs. One row for each interval. The number of
            columns is equal to the number of columns in the input data.
        """
        starts = cuts[:, 0]
        splits = cuts[:, 1]
        ends = cuts[:, 2]
        if self.time_column is None:
            scores = analytical_cont_piecewise_linear_trend_score(
                # scores = linear_trend_score(
                starts=starts,
                splits=splits,
                ends=ends,
                X=self._piecewise_linear_trend_data,
            )
        else:
            scores = lin_reg_cont_piecewise_linear_trend_score(
                starts=starts,
                splits=splits,
                ends=ends,
                X=self._piecewise_linear_trend_data,
                times=self._time_stamps,
            )
        return scores

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate.

        The size of each interval is defined as ``cuts[i, 1] - cuts[i, 0]``.
        To solve for a linear trend, we need at least 2 points.

        TODO: Possible issue, cannot use analytical solution to calculate the score
        when start = split - 1. Need at least one point between start and split.

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
        params = [{}, {}]
        return params
