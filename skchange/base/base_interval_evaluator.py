"""Interval evaluator base class.

    class name: BaseIntervalEvaluator

Scitype defining methods:
    fitting                         - fit(self, X, y=None)
    evaluating                      - evaluate(self, intervals)

Needs to be implemented for a concrete detector:
    _fit(self, X, y=None)
    _evaluate(self, intervals)

Recommended but optional to implement for a concrete detector:
    _check_intervals(self, intervals)
"""

__author__ = ["Tveten"]
__all__ = ["BaseIntervalEvaluator"]

import numpy as np
from numpy.typing import ArrayLike
from sktime.base import BaseEstimator
from sktime.utils.validation.series import check_series

from skchange.utils.validation.data import as_2d_array


class BaseIntervalEvaluator(BaseEstimator):
    """Base class template for interval evaluators.

    This is a common base class for costs, change scores and anomaly scores. It is used
    to evaluate a function on a set of intervals, possibly with split point information.
    """

    def __init__(self):
        self._is_fitted = False
        self._X = None

        super().__init__()

    def fit(self, X, y=None):
        """Fit interval evaluator to training data.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame or np.ndarray
            Data evaluate a function on.
        y : None
            Ignored. Included for API consistency by convention.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        X = check_series(X, allow_index_names=True)
        self._X = X

        self._fit(X=X, y=y)
        self._is_fitted = True
        return self

    def _fit(self, X, y=None):
        """Fit interval evaluator to training data.

        The core logic of fitting a interval evaluator to training data is implemented
        here.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame or np.ndarray
            Data to evaluate a function on.
        y : None
            Ignored. Included for API consistency by convention.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        return self

    def evaluate(self, intervals: ArrayLike) -> np.ndarray:
        """Evaluate on a set of intervals.

        Parameters
        ----------
        intervals : ArrayLike
            Integer location based intervals to evaluate. If intervals is a 2D array,
            the subsets X[intervals[i, 0]:intervals[i, -1]] for
            i = 0, ..., len(intervals) are evaluated.
            If intervals is a 1D array, it is assumed to represent a single interval.

            If intervals contains additional columns, these represent splitting points
            within the interval, and needs to be in increasing order.
            If splitting points are given, the same slicing convention is used for the
            intervals between the splitting points:
            [start, split_1), [split_1, split_2), ..., [split_n, end).

        Returns
        -------
        values : np.ndarray
            One value for each interval.

        Notes
        -----
        Different interval evaluators may require different number of columns in the
        intervals array. For example, a cost function requires two columns, one for the
        start of the interval and one for the end of the interval. A change score
        requires three columns, one for the start of the interval, one for the
        splitting point within the interval, and one for the end of the interval.
        """
        self.check_is_fitted()
        intervals = as_2d_array(intervals, vector_as_column=False)
        intervals = self._check_intervals(intervals)

        values = self._evaluate(intervals)

        if len(values) != len(intervals):
            raise ValueError(f"Expected {len(intervals)} costs, got {len(values)}.")

        return values

    def _evaluate(self, intervals: np.ndarray) -> np.ndarray:
        """Evaluate on a set of intervals.

        The core logic of evaluating a function on a set of intervals is implemented
        here.

        Parameters
        ----------
        intervals : np.ndarray
            A 2D array of integer location based intervals to evaluate.
            The subsets X[intervals[i, 0]:intervals[i, -1]] for
            i = 0, ..., len(intervals) are evaluated.

            If intervals contains additional columns, these represent splitting points
            within the interval, and needs to be in increasing order.
            If splitting points are given, the same slicing convention is used for the
            intervals between the splitting points:
            [start, split_1), [split_1, split_2), ..., [split_n, end).

        Returns
        -------
        values : np.ndarray
            One value for each interval.
        """
        raise NotImplementedError("abstract method")

    def _check_intervals(self, intervals: np.ndarray) -> np.ndarray:
        """Check intervals for compatibility.

        Parameters
        ----------
        intervals : np.ndarray
            A 2D array of integer location based intervals to evaluate.
            the subsets X[intervals[i, 0]:intervals[i, -1]] for
            i = 0, ..., len(intervals) are evaluated.

            If intervals contains additional columns, these represent splitting points
            within the interval, and needs to be in increasing order.
            If splitting points are given, the same slicing convention is used for the
            intervals between the splitting points:
            [start, split_1), [split_1, split_2), ..., [split_n, end).

        Returns
        -------
        intervals : np.ndarray
            The unmodified input intervals array.

        Raises
        ------
        ValueError
            If the intervals are not compatible.
        """
        return intervals
