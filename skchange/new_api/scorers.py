"""Template and examples for implementing interval scorers.

This module demonstrates how to implement custom scorers following sklearn
conventions and the skchange API design.

Base Class
----------
BaseIntervalScorer
    Base class for all interval scorers. Custom scorers must inherit from this
    class and implement the required abstract methods.

Scorer Types
------------
- cost: Objective function for optimization (minimize)
- change_score: Two-sample test for persistent changepoints (maximize)
- saving: One-sample test vs fixed reference (maximize)
- local_saving: One-sample test vs adaptive reference (maximize)

Key Patterns
------------
1. Inherit from BaseIntervalScorer (or type-specific bases like BaseCost)
2. Implement fit() to learn parameters from data (required)
3. Implement evaluate() to score intervals using precomputed data (required)
4. Override precompute() if you need custom preprocessing (optional)
5. Override min_size property if you need minimum interval size > 1 (optional)
6. Override get_default_penalty() to support automatic penalty (optional)
7. Override __sklearn_tags__() to customize tags (optional)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_array, check_is_fitted, validate_data
from typing_extensions import Self

from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils import (
    IntervalScorerTags,
    SkchangeTags,
    check_interval_scorer,
    check_penalty,
)
from skchange.penalties import make_bic_penalty
from skchange.utils.numba import njit
from skchange.utils.numba.stats import col_cumsum


@njit
def l2_cost_optim(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
) -> np.ndarray:
    """Calculate the L2 cost for an optimal constant mean for each segment.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments.
    ends : np.ndarray
        End indices of the segments.
    sums : np.ndarray
        Cumulative sum of the input data, with a row of 0-entries as the first row.
    sums2 : np.ndarray
        Cumulative sum of the squared input data, with a row of 0-entries as the first
        row.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data, where each column
        represents the univariate cost for the corresponding input data column.
    """
    partial_sums = sums[ends] - sums[starts]
    partial_sums2 = sums2[ends] - sums2[starts]
    n = (ends - starts).reshape(-1, 1)
    costs = partial_sums2 - partial_sums**2 / n
    return costs


@njit
def l2_cost_fixed(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """Calculate the L2 cost for a fixed constant mean for each segment.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments.
    ends : np.ndarray
        End indices of the segments.
    sums : np.ndarray
        Cumulative sum of the input data, with a row of 0-entries as the first row.
    sums2 : np.ndarray
        Cumulative sum of the squared input data, with a row of 0-entries as the first
        row.
    mean : np.ndarray
        Fixed mean for the cost calculation.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data, where each column
        represents the univariate cost for the corresponding input data column.
    """
    partial_sums = sums[ends] - sums[starts]
    partial_sums2 = sums2[ends] - sums2[starts]
    n = (ends - starts).reshape(-1, 1)
    costs = partial_sums2 - 2 * mean * partial_sums + n * mean**2
    return costs


@njit
def cusum_score(
    starts: np.ndarray,
    splits: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
) -> np.ndarray:
    """Calculate CUSUM score for change in mean at a split within intervals.

    Parameters
    ----------
    starts : np.ndarray
        Start indices for each interval.
    splits : np.ndarray
        Split indices for each interval.
    ends : np.ndarray
        End indices for each interval.
    sums : np.ndarray
        Cumulative sum of input data with an initial zero row.

    Returns
    -------
    np.ndarray
        Absolute weighted mean differences for each interval and feature.
    """
    n = ends - starts
    before_n = splits - starts
    after_n = ends - splits
    before_sum = sums[splits] - sums[starts]
    after_sum = sums[ends] - sums[splits]
    before_weight = np.sqrt(after_n / (n * before_n)).reshape(-1, 1)
    after_weight = np.sqrt(before_n / (n * after_n)).reshape(-1, 1)
    return np.abs(before_weight * before_sum - after_weight * after_sum)


@njit
def _penalise_scores_constant(scores: np.ndarray, penalty: float) -> np.ndarray:
    """Penalise scores with a constant penalty."""
    return scores.sum(axis=1) - penalty


@njit
def _penalise_scores_linear(
    scores: np.ndarray, penalty_values: np.ndarray
) -> np.ndarray:
    """Penalise scores with linear penalty values."""
    penalty_slope = penalty_values[1] - penalty_values[0]
    penalty_intercept = penalty_values[0] - penalty_slope
    penalised_scores_matrix = (
        np.maximum(scores - penalty_slope, 0.0) - penalty_intercept
    )
    return penalised_scores_matrix.sum(axis=1)


@njit
def _penalise_scores_nonlinear(
    scores: np.ndarray, penalty_values: np.ndarray
) -> np.ndarray:
    """Penalise scores with nonlinear penalty values."""
    penalised_scores = []
    for score in scores:
        sorted_scores = np.sort(score)[::-1]
        penalised_score = np.cumsum(sorted_scores) - penalty_values
        optimal_penalised_score = np.max(penalised_score)
        penalised_scores.append(optimal_penalised_score)
    return np.array(penalised_scores, dtype=np.float64)


class BaseIntervalScorer(BaseEstimator):
    """Base class for interval scorers.

    All interval scorers must inherit from this class and implement the required
    methods. This class provides default implementations for optional methods
    and inherits get_params/set_params from sklearn.base.BaseEstimator.

    Required Methods (must implement)
    ----------------------------------
    fit(X, y=None)
        Learn parameters from training data. Must return self.

    evaluate(precomputed, interval_specs)
        Score intervals using precomputed data. Returns scores array.

    Optional Methods (override as needed)
    --------------------------------------
    precompute(X)
        Validate and precompute statistics. Default validates X and returns dict.

    interval_specs_width
        Expected number of columns in interval_specs. Must override for wrappers
        that delegate to other scorers (e.g., CostBasedChangeScore, PenalisedScore).

    min_size
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
    ...     def evaluate(self, precomputed, interval_specs):
    ...         X = precomputed["X"]
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
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement fit()"
        )

    def evaluate(self, precomputed: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate scorer on intervals.

        Parameters
        ----------
        precomputed : dict
            Precomputed data from precompute().
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

    def precompute(self, X: ArrayLike) -> dict:
        """Default precompute implementation that validates X.

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

    @property
    def interval_specs_width(self) -> int:
        """Expected width of interval specifications for evaluation.

        This property indicates the expected number of columns in the interval_specs
        input to evaluate(). For example, cost scorers typically expect width 2
        (start, end), while change scores expect width 3 (start, split, end).

        In wrappers like CostBasedChangeScore or PenalisedScore, this delegates to the
        wrapped scorer and may require fitting first.

        Returns
        -------
        int
            Expected number of columns in interval_specs for evaluation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement interval_specs_width property. "
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

    This is a convenience base class for cost scorers, which are a common
    type of interval scorer.
    """

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for costs.

        Returns
        -------
        SkchangeTags
            Tags object with default cost configuration.
        """
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.score_type = "cost"
        return tags

    @property
    def interval_specs_width(self) -> int:
        """Expected width of interval specifications for costs.

        Returns
        -------
        int
            Number of columns expected in interval_specs for cost evaluation.
        """
        return 2


class BaseChangeScore(BaseIntervalScorer):
    """Base class for change score scorers.

    This is a convenience base class for change score scorers, which are a common
    type of interval scorer.
    """

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for change scores.

        Returns
        -------
        SkchangeTags
            Tags object with change score configuration.
        """
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.score_type = "change_score"
        return tags

    @property
    def interval_specs_width(self) -> int:
        """Expected width of interval specifications for change scores.

        Returns
        -------
        int
            Number of columns expected in interval_specs for change score evaluation.
        """
        return 3


class BaseSaving(BaseIntervalScorer):
    """Base class for saving scorers.

    This is a convenience base class for saving scorers, which are a common
    type of interval scorer.
    """

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for savings.

        Returns
        -------
        SkchangeTags
            Tags object with saving configuration.
        """
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.score_type = "saving"
        return tags

    @property
    def interval_specs_width(self) -> int:
        """Expected width of interval specifications for saving scores.

        Returns
        -------
        int
            Number of columns expected in interval_specs for saving evaluation.
        """
        return 2


class BaseLocalSaving(BaseIntervalScorer):
    """Base class for local saving scorers.

    This is a convenience base class for local saving scorers, which are a common
    type of interval scorer.
    """

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for local savings.

        Returns
        -------
        SkchangeTags
            Tags object with local saving configuration.
        """
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.score_type = "local_saving"
        return tags

    @property
    def interval_specs_width(self) -> int:
        """Expected width of interval specifications for local saving scores.

        Returns
        -------
        int
            Number of columns expected in interval_specs for local saving evaluation.
        """
        return 4


class L2Cost(BaseCost):
    r"""L2 (squared error) cost function.

    Computes sum of squared deviations from a mean parameter.

    .. math::
        C(X) = \sum_{i=1}^{n} ||x_i - \text{mean}||^2

    Parameters
    ----------
    mean : array-like of shape (n_features,), float, or None, default=None
        Fixed mean parameter. If float, the value is broadcast across all features.
        If None, estimated as sample mean.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Fitted mean parameter.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.scorers import L2Cost
    >>>
    >>> X = np.random.randn(100, 2)
    >>>
    >>> # Estimated mode - learns mean from data
    >>> cost = L2Cost()
    >>> cost.fit(X)
    >>> precomputed = cost.precompute(X)
    >>> interval_specs = np.array([[0, 50], [50, 100]])
    >>> costs = cost.evaluate(precomputed, interval_specs)
    >>>
    >>> # Fixed mode - user provides mean
    >>> cost_fixed = L2Cost(mean=np.array([0.0, 0.0]))
    >>> cost_fixed.fit(X)
    >>> precomputed = cost_fixed.precompute(X)

    Notes
    -----
    This is a simple cost function useful for detecting mean shifts in
    Gaussian data. The estimated version computes the sample mean, while
    the fixed version uses a user-specified reference mean.
    """

    def __init__(self, mean: ArrayLike | float | None = None):
        self.mean = mean

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit L2 cost to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to learn mean from.
        y : None
            Ignored.

        Returns
        -------
        self : L2Cost
            Fitted cost function.
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)  # Sets n_features_in_
        self.n_samples_fit_ = X.shape[0]  # Used for default penalty calculation

        if self.mean is not None:
            if np.isscalar(self.mean):
                mean_arr = np.repeat(self.mean, X.shape[1])
            else:
                mean_arr = check_array(self.mean, ensure_2d=False)
            if mean_arr.shape[0] != X.shape[1]:
                raise ValueError(
                    f"mean must have {X.shape[1]} features, got {mean_arr.shape[0]}"
                )
            self._mean = mean_arr

        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute data for L2 cost evaluation.

        For simple usage, just validates. Could be extended with
        cumulative sums for more efficient evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        precomputed : dict
            Precomputed data with validated array.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        return {
            "sums": col_cumsum(X, init_zero=True),
            "sums2": col_cumsum(X**2, init_zero=True),
        }

    def evaluate(self, precomputed: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate L2 cost on intervals.

        Parameters
        ----------
        precomputed : dict
            Precomputed data from precompute().
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        costs : ndarray of shape (n_interval_specs, n_features)
            L2 costs for each interval and features.
        """
        check_is_fitted(self)

        interval_specs = check_array(
            interval_specs,
            ensure_2d=True,
            ensure_min_features=self.interval_specs_width,
        )
        starts = interval_specs[:, 0]
        ends = interval_specs[:, 1]

        sums, sums2 = precomputed["sums"], precomputed["sums2"]
        if self.mean is None:
            costs = l2_cost_optim(starts, ends, sums, sums2)
        else:
            costs = l2_cost_fixed(starts, ends, sums, sums2, self._mean)

        return costs

    def get_default_penalty(self) -> float:
        """Get default penalty value for L2 cost.

        Returns
        -------
        float
            Default penalty value for L2 cost.
        """
        check_is_fitted(self)
        return make_bic_penalty(self.n_features_in_, self.n_samples_fit_)


class CUSUM(BaseChangeScore):
    """CUSUM change score for a change in mean.

    Computes the classical CUSUM statistic as the weighted absolute difference
    between the means before and after a split point within each interval.

    Notes
    -----
    Unlike `L2Cost`, this scorer has no fixed-vs-optim parameter variants,
    so no mode dispatch logic is required.
    """

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit CUSUM scorer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Used to validate shape and feature count.
        y : None
            Ignored.

        Returns
        -------
        self : CUSUM
            Fitted scorer.
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)  # Sets n_features_in_
        self.n_samples_fit_ = X.shape[0]  # Used for default penalty calculation
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute cumulative sums for CUSUM evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        precomputed : dict
            Dictionary with cumulative sums under key ``"sums"``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        return {"sums": col_cumsum(X, init_zero=True)}

    def evaluate(self, precomputed: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate CUSUM score at splits within intervals.

        Parameters
        ----------
        precomputed : dict
            Precomputed data from precompute().
        interval_specs : array-like of shape (n_interval_specs, 3)
            Interval boundaries and split locations ``[start, split, end)``.

        Returns
        -------
        scores : ndarray of shape (n_interval_specs, n_features)
            CUSUM scores for each interval specification and feature.
        """
        check_is_fitted(self)
        sums = precomputed["sums"]

        interval_specs = check_array(
            interval_specs,
            ensure_2d=True,
            ensure_min_features=self.interval_specs_width,
        )
        if interval_specs.shape[1] != self.interval_specs_width:
            raise ValueError(
                f"interval_specs must have shape"
                f" (n_interval_specs, {self.interval_specs_width}), "
                f"got {interval_specs.shape}."
            )

        starts = interval_specs[:, 0]
        splits = interval_specs[:, 1]
        ends = interval_specs[:, 2]

        if np.any(splits <= starts) or np.any(splits >= ends):
            raise ValueError(
                "Each interval specification must satisfy start < split < end "
                "for CUSUM evaluation."
            )

        return cusum_score(starts, splits, ends, sums)

    def get_default_penalty(self) -> float:
        """Get default penalty value for the fitted CUSUM score."""
        bic_penalty = make_bic_penalty(self.n_features_in_, self.n_samples_fit_)
        # BIC works on a squared error scale, while CUSUM is on an absolute error scale.
        return np.sqrt(bic_penalty)

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for CUSUM scorer."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.score_type = "change_score"
        return tags


class CostBasedChangeScore(BaseChangeScore):
    """Change scorer constructed from a cost scorer.

    Computes change score as the cost reduction of allowing a split within an interval:

    ``score = cost(start, end) - cost(start, split) - cost(split, end)``
    """

    def __init__(self, cost: BaseIntervalScorer):
        self.cost = cost

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit wrapped cost scorer."""
        X = validate_data(self, X, ensure_2d=True, reset=True)
        self.cost_: BaseIntervalScorer = clone(self.cost).fit(X, y)
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute wrapped cost scorer data."""
        check_is_fitted(self, ["cost_"])
        return {"cost_precomputed": self.cost_.precompute(X)}

    def evaluate(self, precomputed: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate change score on interval specifications.

        Parameters
        ----------
        precomputed : dict
            Precomputed data from precompute().
        interval_specs : array-like of shape (n_interval_specs, 3)
            Interval boundaries and split locations ``[start, split, end)``.

        Returns
        -------
        scores : ndarray of shape (n_interval_specs,) or (n_interval_specs, n_features)
            Change scores for each interval specification.
        """
        check_is_fitted(self, ["cost_"])
        interval_specs = check_array(interval_specs, ensure_2d=True, dtype=np.int64)
        if interval_specs.shape[1] != 3:
            raise ValueError(
                "interval_specs must have shape (n_interval_specs, 3), "
                f"got {interval_specs.shape}."
            )

        starts = interval_specs[:, 0]
        splits = interval_specs[:, 1]
        ends = interval_specs[:, 2]
        if np.any(splits <= starts) or np.any(splits >= ends):
            raise ValueError(
                "Each interval specification must satisfy start < split < end "
                "for change-score evaluation."
            )

        cost_precomputed = precomputed["cost_precomputed"]
        left_intervals = interval_specs[:, [0, 1]]
        right_intervals = interval_specs[:, [1, 2]]
        full_intervals = interval_specs[:, [0, 2]]

        left_costs = self.cost_.evaluate(cost_precomputed, left_intervals)
        right_costs = self.cost_.evaluate(cost_precomputed, right_intervals)
        no_change_costs = self.cost_.evaluate(cost_precomputed, full_intervals)

        change_scores = no_change_costs - (left_costs + right_costs)
        change_scores[(change_scores < 0) & (change_scores > -1e-8)] = 0.0
        return change_scores

    @property
    def min_size(self) -> int:
        """Minimum valid interval size inherited from wrapped cost scorer."""
        check_is_fitted(self)
        return self.cost_.min_size

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for change score wrapper."""
        tags = super().__sklearn_tags__()
        cost_tags = self.cost.__sklearn_tags__().interval_scorer_tags
        tags.interval_scorer_tags.score_type = "change_score"
        tags.interval_scorer_tags.conditional = cost_tags.conditional
        tags.interval_scorer_tags.aggregated = cost_tags.aggregated
        tags.interval_scorer_tags.penalised = cost_tags.penalised
        return tags


class PenalisedScore(BaseIntervalScorer):
    """Penalised interval scorer wrapper for new API scorers.

    Aggregates feature-wise scores and applies either constant, linear, or
    nonlinear penalties over the number of affected variables.

    Let sorted_score be the k-th largest feature-wise score for an interval, then
    the penalised score is computed as
    ``np.max(np.cumsum(sorted_scores) - penalty_values)``.

    For a constant penalty and a linear penalty, this reduces to simpler forms that can
    be computed more efficiently.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        The base interval scorer to wrap and penalise.
    penalty : array-like of shape (n_features,), float, or None, default=None
        Penalty values. `penalty[k]` is the penalty for including k features in the
        aggregated penalised score. If float, the value is broadcast across all k.

    """

    def __init__(
        self,
        scorer: BaseIntervalScorer,
        penalty: ArrayLike | float | None = None,
    ):
        self.scorer = scorer
        self.penalty = penalty

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit wrapped scorer and select penalty mode."""
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            dtype=np.float64,
            reset=True,
        )

        self.scorer_ = check_interval_scorer(
            self.scorer,
            required_tasks=["change_score", "saving", "local_saving"],
            allow_penalised=False,
            clone=True,
            caller_name=self.__class__.__name__,
            arg_name="scorer",
        )
        self.scorer_.fit(X, y)

        scorer_tags = self.scorer_.__sklearn_tags__().interval_scorer_tags
        if scorer_tags.aggregated and self.penalty is not None:
            penalty_arr = np.asarray(self.penalty).reshape(-1)
            if penalty_arr.size > 1:
                raise ValueError(
                    "`penalty` must be scalar for aggregated input scores."
                )

        penalty = (
            self.scorer_.get_default_penalty() if self.penalty is None else self.penalty
        )
        self.penalty_ = check_penalty(
            penalty,
            caller_name=self.__class__.__name__,
            arg_name="penalty",
        )

        penalty_values = np.asarray(self.penalty_).reshape(-1)
        if penalty_values.size > 1 and penalty_values.size != X.shape[1]:
            raise ValueError(
                "`penalty` must be scalar or have length equal to n_features. "
                f"Got penalty length {penalty_values.size} and n_features {X.shape[1]}."
            )

        if X.shape[1] == 1 or penalty_values.size == 1:
            self._penalty_mode = "constant"
        elif np.allclose(np.diff(penalty_values), np.diff(penalty_values)[0]):
            self._penalty_mode = "linear"
        else:
            self._penalty_mode = "nonlinear"

        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute wrapped scorer data for penalised evaluation."""
        check_is_fitted(self, ["scorer_", "penalty_", "_penalty_mode"])
        scorer_precomputed = self.scorer_.precompute(X)
        return {"scorer_precomputed": scorer_precomputed}

    def evaluate(self, precomputed: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate penalised scores on interval specifications.

        Parameters
        ----------
        precomputed : dict
            Precomputed data from precompute().
        interval_specs : array-like
            Each row specifies an interval and possibly split points, depending on
            the wrapped scorer type. The expected shape is determined by
            ``self.scorer_``.

        Returns
        -------
        scores : ndarray of shape (n_interval_specs, 1)
            Penalised, aggregated score for each interval specification.
        """
        check_is_fitted(self, ["scorer_", "penalty_", "_penalty_mode"])

        scores = self.scorer_.evaluate(
            precomputed["scorer_precomputed"],
            interval_specs,
        )
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)

        if self._penalty_mode == "constant":
            penalty_value = float(np.asarray(self.penalty_).reshape(-1)[0])
            penalised = _penalise_scores_constant(scores, penalty_value)
        elif self._penalty_mode == "linear":
            penalty_values = np.asarray(self.penalty_, dtype=np.float64).reshape(-1)
            penalised = _penalise_scores_linear(scores, penalty_values)
        elif self._penalty_mode == "nonlinear":
            penalty_values = np.asarray(self.penalty_, dtype=np.float64).reshape(-1)
            penalised = _penalise_scores_nonlinear(scores, penalty_values)
        else:
            raise RuntimeError(f"Unknown penalty mode: {self._penalty_mode}")

        return penalised.reshape(-1, 1)

    @property
    def interval_specs_width(self) -> int:
        """Expected width of interval specifications inherited from wrapped scorer."""
        check_is_fitted(self)
        return self.scorer_.interval_specs_width

    @property
    def min_size(self) -> int:
        """Minimum valid interval size inherited from wrapped scorer."""
        check_is_fitted(self)
        return self.scorer_.min_size

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for penalised scorer."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.score_type = (
            self.scorer.__sklearn_tags__().interval_scorer_tags.score_type
        )
        tags.interval_scorer_tags.aggregated = True
        tags.interval_scorer_tags.penalised = True
        return tags
