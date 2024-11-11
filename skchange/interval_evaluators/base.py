"""Base classes for detection scores."""

__author__ = ["Tveten"]

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

    def _check_intervals(self, intervals):
        """Check intervals for compatibility."""
        raise NotImplementedError("abstract method")

    def evaluate(self, intervals) -> float:
        """Evaluate on an interval."""
        self.check_is_fitted()
        intervals = self._check_intervals(intervals)

        value = self._evaluate(intervals)
        return value

    def _evaluate(self, intervals) -> float:
        raise NotImplementedError("abstract method")