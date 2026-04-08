"""Base classes and type-checking utilities for interval scorers.

Base Classes
------------
BaseIntervalScorer
    Base class for all interval scorers.
BaseCost
    Base class for cost scorers (score_type='cost').
BaseChangeScore
    Base class for change score scorers (score_type='change_score').
BaseSaving
    Base class for saving scorers (score_type='saving').
BaseTransientScore
    Base class for transient score scorers (score_type='transient_score').

Type-checking Utilities
-----------------------
is_cost, is_change_score, is_saving, is_transient_score
    Return True if an estimator is of the given interval scorer type.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import get_tags
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.typing import ArrayLike, Self
from skchange.new_api.utils._tags import IntervalScorerTags, SkchangeTags
from skchange.new_api.utils.validation import validate_data
from skchange.penalties import make_bic_penalty


class BaseIntervalScorer(BaseEstimator):
    """Base class for interval scorers.

    All interval scorers must inherit from this class and implement the required
    methods. This class provides default implementations for optional methods
    and inherits get_params/set_params from sklearn.base.BaseEstimator.

    Required Methods (must implement)
    ----------------------------------
    fit(X, y=None)
        Learn parameters from training data. Must return self.

    evaluate(cache, interval_specs)
        Score intervals using precomputed data. Returns scores array.

    Optional Methods (override as needed)
    --------------------------------------
    precompute(X)
        Validate and precompute statistics. Default validates X and returns dict.

    interval_specs_ncols
        Expected number of columns in interval_specs. Must override for wrappers
        that delegate to other scorers (e.g., CostChangeScore, PenalisedScore).
        Minimum valid interval size. Default returns 1.

    get_default_penalty()
        Default penalty for automatic penalty selection. Override to support this.

    __sklearn_tags__()
        Scorer metadata (score_type, conditional, aggregated, penalised).
        Override to customize tags.

    Examples
    --------
    >>> class MyScorer(BaseIntervalScorer):
    ...     def __sklearn_tags__(self):
    ...         tags = super().__sklearn_tags__()
    ...         tags.interval_scorer_tags.score_type = "change_score"
    ...         return tags
    ...
    ...     def fit(self, X, y=None):
    ...         X = validate_data(self, X, ensure_2d=True, reset=True)
    ...         self.threshold_ = np.std(X)
    ...         return self
    ...
    ...     def evaluate(self, cache, interval_specs):
    ...         X = cache["X"]
    ...         interval_specs = np.asarray(interval_specs)
    ...         scores = [
    ...             np.abs(np.mean(X[s:e])) / self.threshold_ for s, e in interval_specs
    ...         ]
    ...         return np.array(scores)

    Notes
    -----
    This class follows sklearn conventions. Fitted attributes end with underscore.
    """

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the scorer to training data.

        The default implementation validates ``X`` and stores ``n_features_in_``
        and ``n_samples_in_``. Override when the scorer needs to learn parameters
        from the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like or None
            Ignored. Exists for API compatibility.

        Returns
        -------
        self
            Fitted scorer instance.
        """
        validate_data(self, X, ensure_2d=True, reset=True)
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute statistics from data to speed up evaluate.

        Defaults to validating X and returning it in a dict. Override to precompute
        specific statistics needed for evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        dict
            Dictionary containing the validated array under key 'X'.
        """
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            dtype=np.float64,
            reset=False,
        )
        return {"X": X}

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate scorer on intervals.

        Parameters
        ----------
        cache : dict
            Cache from precompute().
        interval_specs : 2d array-like
            Each row specifies an interval and possibly split points, depending on
            the scorer type. For example, shape (n_interval_specs, 2) for cost scorers
            with columns [start, end), and shape (n_interval_specs, 3) for change scores
            with columns (start, split, end), where the data is split into
            [start, split) and [split, end). The expected format of interval_specs
            is documented by each scorer implementation.

        Returns
        -------
        scores : ndarray of shape (n_interval_specs,) or (n_interval_specs, n_features)
            Scores for each interval. If the scorer computes aggregated scores across
            features, the output shape is (n_interval_specs,). If the scorer computes
            feature-wise scores, the output shape is (n_interval_specs, n_features).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluate()"
        )

    @property
    def interval_specs_ncols(self) -> int:
        """Expected number of columns in interval_specs for evaluation.

        For example, cost scorers typically expect 2 columns (start, end), while
        change scores expect 3 (start, split, end).

        In wrappers like CostChangeScore or PenalisedScore, this delegates to the
        wrapped scorer and may require fitting first.

        Returns
        -------
        int
            Expected number of columns in interval_specs.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement interval_specs_ncols property. "
            f"Type-specific base classes (BaseCost, BaseChangeScore, etc.) provide "
            f"this automatically."
        )

    @property
    def min_size(self) -> int:
        """Minimum valid size of an interval to evaluate.

        ``np.diff(interval_specs, axis=1)`` must be ``>= min_size`` for valid
        evaluation.

        Returns
        -------
        int
            Minimum interval size.
        """
        return 1

    def get_default_penalty(self) -> float:
        """Get default penalty value for L2 cost.

        Returns
        -------
        float
            Default penalty value for L2 cost.
        """
        check_is_fitted(self)
        return make_bic_penalty(self.n_features_in_, self.n_samples_in_)

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for the interval scorer.

        Returns
        -------
        SkchangeTags
            Tags object with interval scorer specific configuration.

        Notes
        -----
        Subclasses should override this to customize tags.
        """
        tags = SkchangeTags()
        tags.interval_scorer_tags = IntervalScorerTags()
        return tags


class BaseCost(BaseIntervalScorer):
    """Base class for cost scorers.

    Convenience base class that sets ``score_type='cost'`` and
    ``interval_specs_ncols=2``. Subclasses receive interval specs with columns
    ``[start, end)``.
    """

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags with ``score_type='cost'``."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.score_type = "cost"
        return tags

    @property
    def interval_specs_ncols(self) -> int:
        """Return 2 (columns: start, end)."""
        return 2


class BaseChangeScore(BaseIntervalScorer):
    """Base class for change score scorers.

    Convenience base class that sets ``score_type='change_score'`` and
    ``interval_specs_ncols=3``. Subclasses receive interval specs with columns
    ``[start, split, end)``.
    """

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags with ``score_type='change_score'``."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.score_type = "change_score"
        return tags

    @property
    def interval_specs_ncols(self) -> int:
        """Return 3 (columns: start, split, end)."""
        return 3


class BaseSaving(BaseIntervalScorer):
    """Base class for saving scorers.

    Convenience base class that sets ``score_type='saving'`` and
    ``interval_specs_ncols=2``. Subclasses receive interval specs with columns
    ``[start, end)``.
    """

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags with ``score_type='saving'``."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.score_type = "saving"
        return tags

    @property
    def interval_specs_ncols(self) -> int:
        """Return 2 (columns: start, end)."""
        return 2


class BaseTransientScore(BaseIntervalScorer):
    """Base class for transient score scorers.

    Convenience base class that sets ``score_type='transient_score'`` and
    ``interval_specs_ncols=4``. Subclasses receive interval specs with four columns
    representing a segment that deviates from a reference (background) distribution
    and then returns to it — the epidemic changepoint model.
    """

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags with ``score_type='transient_score'``."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.score_type = "transient_score"
        return tags

    @property
    def interval_specs_ncols(self) -> int:
        """Return 4."""
        return 4


def is_cost(estimator) -> bool:
    """Return True if the given estimator is (probably) a cost scorer.

    Parameters
    ----------
    estimator : estimator instance
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a cost scorer and False otherwise.

    Examples
    --------
    >>> from skchange.new_api.interval_scorers import L2Cost, CUSUM
    >>> is_cost(L2Cost())
    True
    >>> is_cost(CUSUM())
    False
    """
    scorer_tags = get_tags(estimator).interval_scorer_tags  # type: ignore[union-attr]
    return scorer_tags is not None and scorer_tags.score_type == "cost"


def is_change_score(estimator) -> bool:
    """Return True if the given estimator is (probably) a change score scorer.

    Parameters
    ----------
    estimator : estimator instance
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a change score scorer and False otherwise.

    Examples
    --------
    >>> from skchange.new_api.interval_scorers import L2Cost, CUSUM
    >>> is_change_score(CUSUM())
    True
    >>> is_change_score(L2Cost())
    False
    """
    scorer_tags = get_tags(estimator).interval_scorer_tags  # type: ignore[union-attr]
    return scorer_tags is not None and scorer_tags.score_type == "change_score"


def is_saving(estimator) -> bool:
    """Return True if the given estimator is (probably) a saving scorer.

    Parameters
    ----------
    estimator : estimator instance
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a saving scorer and False otherwise.
    """
    scorer_tags = get_tags(estimator).interval_scorer_tags  # type: ignore[union-attr]
    return scorer_tags is not None and scorer_tags.score_type == "saving"


def is_transient_score(estimator) -> bool:
    """Return True if the given estimator is (probably) a transient score scorer.

    Parameters
    ----------
    estimator : estimator instance
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a transient score scorer and False otherwise.
    """
    scorer_tags = get_tags(estimator).interval_scorer_tags  # type: ignore[union-attr]
    return scorer_tags is not None and scorer_tags.score_type == "transient_score"


def is_penalised_score(estimator) -> bool:
    """Return True if the given estimator is a penalised interval scorer.

    Parameters
    ----------
    estimator : estimator instance
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is penalised and False otherwise.

    Examples
    --------
    >>> from skchange.new_api.interval_scorers import CUSUM, PenalisedScore
    >>> is_penalised_score(PenalisedScore(CUSUM()))
    True
    >>> is_penalised_score(CUSUM())
    False
    """
    scorer_tags = get_tags(estimator).interval_scorer_tags  # type: ignore[union-attr]
    return scorer_tags is not None and scorer_tags.penalised
