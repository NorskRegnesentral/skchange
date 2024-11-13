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

from sktime.base import BaseEstimator
from sktime.utils.validation.series import check_series


class BaseIntervalEvaluator(BaseEstimator):
    """Base class template for interval evaluators.

    Interval evaluators are used to evaluate a function on given intervals of data.
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

    def evaluate(self, intervals) -> float:
        """Evaluate on a set of intervals."""
        self.check_is_fitted()
        intervals = self._check_intervals(intervals)

        values = self._evaluate(intervals)

        if len(values) != len(intervals):
            raise ValueError(f"Expected {len(intervals)} costs, got {len(values)}.")

        return values

    def _evaluate(self, intervals) -> float:
        raise NotImplementedError("abstract method")

    def _check_intervals(self, intervals):
        """Check intervals for compatibility."""
        return intervals
