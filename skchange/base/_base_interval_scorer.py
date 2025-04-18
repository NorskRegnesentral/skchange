"""Interval scorer base class.

    class name: BaseIntervalScorer

Scitype defining methods:
    fitting                         - fit(self, X, y=None)
    evaluating                      - evaluate(self, cuts)

Needs to be implemented for a concrete detector:
    _fit(self, X, y=None)
    _evaluate(self, cuts)
"""

__author__ = ["Tveten", "johannvk", "fkiraly"]
__all__ = ["BaseIntervalScorer"]

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sktime.base import BaseEstimator
from sktime.utils.validation.series import check_series

from skchange.utils.validation.cuts import check_cuts_array
from skchange.utils.validation.data import as_2d_array
from skchange.utils.validation.enums import EvaluationType


class BaseIntervalScorer(BaseEstimator):
    """Base class template for interval scorers.

    This is a common base class for costs, change scores, and anomaly scores. It is used
    as a building block for detectors in skchange. The class provides a common interface
    to evaluate a scoring function on a set of data cuts. Depending on the sub class,
    the cuts may represent either intervals to subset or splits within intervals.

    Attributes
    ----------
    _is_fitted : bool
        Indicates whether the interval scorer has been fitted.
    _X : np.ndarray
        The data input to `fit` coerced to a 2D ``np.ndarray``.
    """

    _tags = {
        "object_type": "interval_scorer",  # type of object
        "authors": ["Tveten", "johannvk", "fkiraly"],  # author(s) of the object
        "maintainers": "Tveten",  # current maintainer(s) of the object
    }  # for unit test cases

    # Number of expected entries in the cuts array of `evaluate`. Must be overridden in
    # subclasses, like for change scores and costs, before call to `evaluate`.
    expected_cut_entries = None

    # The `evaluation_type` indicates whether the scorer is univariate or multivariate.
    # Univariate scorers are vectorized over variables/columns in the data,
    # such that output is one column per variable.
    # Multivariate scorers take the entire data as input and output a single
    # value, such that the output is a single column no matter how many variables.
    # TODO: Implement as tags?
    # For now a class variable to pass sktime conformance test.
    evaluation_type = EvaluationType.UNIVARIATE

    # `is_penalised_score` indicates whether the score is inherently penalised (True) or
    # not (False).
    # For example, for change scores or savings:
    # If `True`, a score > 0 means that a change or anomaly is detected. Penalised
    # scores can be both positive and negative.
    # If `False`, the score is not penalised. To test for the existence of a change,
    # penalisation must be performed externally. Such scores are always non-negative.
    is_penalised_score = False

    def __init__(self):
        self._is_fitted = False
        self._X = None

        super().__init__()

    def fit(self, X, y=None):
        """Fit the interval scorer to the training data.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, or np.ndarray
            Data to evaluate.
        y : None
            Ignored. Included for API consistency by convention.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates the fitted model and sets attributes ending in ``"_"``.
        """
        X = check_series(X, allow_index_names=True)
        if isinstance(X, pd.DataFrame):
            self._X_columns = X.columns
        else:
            self._X_columns = None
        self._X = as_2d_array(X)

        self._fit(X=self._X, y=y)
        self._is_fitted = True
        return self

    def _fit(self, X: np.ndarray, y=None):
        """Fit the interval scorer to training data.

        The core logic of fitting an interval scorer to training data is implemented
        here.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y : None
            Ignored. Included for API consistency by convention.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in ``"_"``.
        """
        return self

    def evaluate(self, cuts: ArrayLike) -> np.ndarray:
        """Evaluate the score according to a set of cuts.

        Parameters
        ----------
        cuts : ArrayLike
            A 2D array of integer location-based cuts to evaluate where each row gives
            a single cut specification. If a 1D array is passed, it is assumed to
            be a row vector. Each cut divide the data into one or more intervals for
            evaluation and may contain multiple entries representing for example the
            start and end of the interval, and potentially split points within the
            interval. Each cut must be sorted in increasing order. Subclasses specify
            the further expected structure of the cuts array and how it is used
            internally to evaluate the score.

        Returns
        -------
        scores : np.ndarray
            A 2D array of scores. One row for each row in cuts.
        """
        self.check_is_fitted()
        cuts = as_2d_array(cuts, vector_as_column=False)
        cuts = self._check_cuts(cuts)

        values = self._evaluate(cuts)
        return values

    def _evaluate(self, cuts: np.ndarray) -> np.ndarray:
        """Evaluate the score on a set of cuts.

        The core logic of evaluating a function on according to the cuts is implemented
        here.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array of integer location-based cuts to evaluate. Each row in the array
            must be sorted in increasing order.

        Returns
        -------
        values : np.ndarray
            A 2D array of scores. One row for each row in cuts.
        """
        raise NotImplementedError("abstract method")

    @property
    def min_size(self) -> int | None:
        """Minimum valid size of an interval to evaluate.

        The size of each interval is by default defined as ``np.diff(cuts[i, ])``.
        Subclasses can override the min_size to mean something else, for example in
        cases where intervals are combined before evaluation or `cuts` specify
        disjoint intervals.

        Returns
        -------
        int or None
            The minimum valid size of an interval to evaluate. If ``None``, it is
            unknown what the minimum size is. E.g., the scorer may need to be fitted
            first to determine the minimum size.
        """
        return 1

    @property
    def output_dim(self) -> int:
        """Get the output dimension of the scorer.

        Returns the number of columns in the output of the scorer. For univariate
        scorers, this is the number of variables in the data. For multivariate scorers,
        this is 1. Subclasses should override this method accordingly.
        """
        if self.evaluation_type == EvaluationType.UNIVARIATE:
            return self._X.shape[1] if self._X is not None else None
        elif self.evaluation_type == EvaluationType.MULTIVARIATE:
            return 1
        elif self.evaluation_type == EvaluationType.CONDITIONAL:
            return self._output_dim()

    def _output_dim(self) -> int:
        """Get the output dimension of the scorer.

        In the case of a scorer with 'evaluation_type' as 'CONDITIONAL', this method
        should be overridden to return the number of columns in the scorer output.
        Subclasses should override this method accordingly.
        """
        raise NotImplementedError("abstract method")

    def get_param_size(self, p: int) -> int:
        """Get the number of parameters to estimate over each interval.

        The primary use of this method is to determine an appropriate default penalty
        value in detectors. For example, a scorer for a change in mean has one
        parameter to estimate per variable in the data, a scorer for a change in the
        mean and variance has two parameters to estimate per variable, and so on.
        Subclasses should override this method accordingly.

        Parameters
        ----------
        p : int
            Number of variables in the data.
        """
        return p

    def _check_cuts(self, cuts: np.ndarray) -> np.ndarray:
        """Check cuts for compatibility.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array of integer location-based cuts to evaluate. Each row in the array
            must be sorted in increasing order.

        Returns
        -------
        cuts : np.ndarray
            The unmodified input `cuts` array.

        Raises
        ------
        ValueError
            If the `cuts` are not compatible.
        """
        return check_cuts_array(
            cuts,
            min_size=self.min_size,
            last_dim_size=self.expected_cut_entries,
        )

    def check_is_penalised(self):
        """Check if the score is inherently performing penalisation."""
        if not self.is_penalised_score:
            raise ValueError("The score is not penalised.")
