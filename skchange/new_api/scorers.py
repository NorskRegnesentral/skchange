"""Template and examples for implementing interval scorers.

This module demonstrates how to implement custom scorers following sklearn
conventions and the skchange API design.

Protocol
--------
IntervalScorer : Protocol
    Defines the interface for all interval scorers. Use for type hints
    and polymorphic detector code. No inheritance required.

Scorer Types
------------
- cost: Objective function for optimization (minimize)
- change_score: Two-sample test for persistent changepoints (maximize)
- saving: One-sample test vs fixed reference (maximize)
- local_saving: One-sample test vs adaptive reference (maximize)

Key Patterns
------------
1. List model parameters in _model_params
2. Use None to indicate "estimate mode" for model parameters
3. Implement fit() to learn parameters from data
4. Implement precompute() to validate and precompute (mandatory)
5. Implement evaluate() to score intervals using precomputed data
6. Implement min_size property for minimum interval size
7. Implement get_default_penalty() for automatic penalty calculation
8. Override __sklearn_tags__() to customize tags
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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


@runtime_checkable
class IntervalScorer(Protocol):
    """Protocol for interval scorers.

    Defines the interface that all interval scorers must implement. Classes
    implementing this protocol can be used polymorphically in detectors without
    requiring inheritance.

    Required Methods
    ----------------
    fit(X, y=None)
        Learn model parameters from training data.

    precompute(X)
        Validate and precompute statistics for efficient evaluation.

    evaluate(precomputed, cuts)
        Score intervals using precomputed data.

    min_size
        Property returning minimum valid interval size.

    get_default_penalty(n_samples, n_features)
        Calculate default penalty for automatic penalty selection.

    Examples
    --------
    >>> def process_with_scorer(scorer: IntervalScorer, X, cuts):
    ...     scorer.fit(X)
    ...     precomputed = scorer.precompute(X)
    ...     return scorer.evaluate(precomputed, cuts)
    >>>
    >>> # Any class implementing the protocol works
    >>> from skchange.new_api.scorers import L2Cost
    >>> cost = L2Cost()
    >>> isinstance(cost, IntervalScorer)  # True with @runtime_checkable
    True

    Notes
    -----
    The @runtime_checkable decorator enables isinstance() checks at runtime,
    though this only verifies method existence, not signatures.
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
        ...

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute data for evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute for evaluation.

        Returns
        -------
        precomputed : dict
            Dictionary containing at minimum the validated array under key 'X'.
            May include additional precomputations for efficiency.
        """
        ...

    def evaluate(self, precomputed: dict, cuts: ArrayLike) -> np.ndarray:
        """Evaluate scorer on intervals.

        Parameters
        ----------
        precomputed : dict
            Precomputed data from precompute().
        cuts : array-like of shape (n_cuts, 2)
            Interval boundaries [start, end) to score.

        Returns
        -------
        scores : ndarray of shape (n_cuts,) or (n_cuts, n_features)
            Scores for each interval.
        """
        ...

    @property
    def min_size(self) -> int:
        """Minimum valid size of an interval to evaluate.

        In general, min_size cannot be resolved until after fitting.

        Returns
        -------
        int
            Minimum interval size.
        """
        ...

    def get_default_penalty(self, n_samples: int, n_features: int) -> float:
        """Get default penalty value.

        In general, default penalty cannot be resolved until after fitting.

        Parameters
        ----------
        n_samples : int
            Number of samples in the data.
        n_features : int
            Number of features in the data.

        Returns
        -------
        penalty : float
            Default penalty value.
        """
        ...


class BaseIntervalScorer(BaseEstimator):
    """Base class for interval scorers with sensible defaults.

    Provides default implementations for min_size and get_default_penalty,
    inherits get_params/set_params from sklearn.base.BaseEstimator.

    Subclasses must implement:
    - fit(X, y=None)
    - precompute(X)
    - evaluate(precomputed, cuts)

    Subclasses should override:
    - __sklearn_tags__() to customize tags

    Examples
    --------
    >>> class MyScorer(BaseIntervalScorer):
    ...     def __sklearn_tags__(self):
    ...         tags = super().__sklearn_tags__()
    ...         tags.interval_scorer_tags.score_type = "change_score"
    ...         return tags
    ...
    ...     def fit(self, X, y=None):
    ...         X = check_array(X)
    ...         self.threshold_ = np.std(X)
    ...         return self
    ...
    ...     def precompute(self, X):
    ...         check_is_fitted(self)
    ...         X = check_array(X)
    ...         return {"X": X}
    ...
    ...     def evaluate(self, precomputed, cuts):
    ...         X = precomputed["X"]
    ...         cuts = np.asarray(cuts)
    ...         scores = [np.abs(np.mean(X[s:e])) / self.threshold_ for s, e in cuts]
    ...         return np.array(scores)

    Notes
    -----
    Inheriting from this base class is optional - you can implement the
    IntervalScorer protocol directly. This class provides convenience
    for common patterns.
    """

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

    @property
    def min_size(self) -> int:
        """Minimum valid size of an interval to evaluate.

        Returns
        -------
        int
            Minimum interval size. Default is 1.

        Notes
        -----
        Override this in subclasses if a larger minimum size is required
        (e.g., for computing statistics that require multiple samples).
        """
        return 1

    def get_default_penalty(self, n_samples: int, n_features: int) -> float:
        """Get default penalty value.

        Uses BIC-style penalty: k * log(n) where k is the number of
        model parameters per feature.

        Parameters
        ----------
        n_samples : int
            Number of samples in the data.
        n_features : int
            Number of features in the data.

        Returns
        -------
        penalty : float
            Default penalty value based on model complexity.

        Notes
        -----
        Override this in subclasses for scorer-specific penalty calculations.
        The default assumes one parameter per feature.
        """
        # Default: assume one model parameter per feature
        n_params = n_features
        return make_bic_penalty(n_params, n_samples)


class L2Cost(BaseIntervalScorer):
    r"""L2 (squared error) cost function.

    Computes sum of squared deviations from a mean parameter.

    .. math::
        C(X) = \sum_{i=1}^{n} ||x_i - \mu||^2

    Parameters
    ----------
    mu : array-like of shape (n_features,) or None, default=None
        Fixed mean parameter. If None, estimated as sample mean.

    Attributes
    ----------
    mu_ : ndarray of shape (n_features,)
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
    >>> cuts = np.array([[0, 50], [50, 100]])
    >>> costs = cost.evaluate(precomputed, cuts)
    >>>
    >>> # Fixed mode - user provides mean
    >>> cost_fixed = L2Cost(mu=np.array([0.0, 0.0]))
    >>> cost_fixed.fit(X)
    >>> precomputed = cost_fixed.precompute(X)

    Notes
    -----
    This is a simple cost function useful for detecting mean shifts in
    Gaussian data. The estimated version computes the sample mean, while
    the fixed version uses a user-specified reference mean.
    """

    _model_params = ["mu"]

    def __init__(self, mu: ArrayLike | None = None):
        self.mu = mu

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
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            dtype=np.float64,
            reset=True,
        )

        if self.mu is None:
            self.mu_ = None
            self._eval_mode = "optim"
        else:
            mu_arr = check_array(
                self.mu,
                ensure_2d=False,
                dtype=np.float64,
            )
            if mu_arr.shape[0] != X.shape[1]:
                raise ValueError(
                    f"mu must have {X.shape[1]} features, got {mu_arr.shape[0]}"
                )
            self.mu_ = mu_arr
            self._eval_mode = "fixed"

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
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            dtype=np.float64,
            reset=False,
        )
        return {
            "sums": col_cumsum(X, init_zero=True),
            "sums2": col_cumsum(X**2, init_zero=True),
        }

    def evaluate(self, precomputed: dict, cuts: ArrayLike) -> np.ndarray:
        """Evaluate L2 cost on intervals.

        Parameters
        ----------
        precomputed : dict
            Precomputed data from precompute().
        cuts : array-like of shape (n_cuts, 2)
            Interval boundaries [start, end).

        Returns
        -------
        costs : ndarray of shape (n_cuts,)
            L2 cost for each interval (summed across features).
        """
        check_is_fitted(self, ["mu_", "_eval_mode"])

        sums = precomputed["sums"]
        sums2 = precomputed["sums2"]

        cuts = check_array(cuts, ensure_2d=True, dtype=np.int64)
        starts = cuts[:, 0]
        ends = cuts[:, 1]

        costs = self._evaluate_dispatch(starts, ends, sums, sums2)
        return costs

    def _evaluate_dispatch(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        sums: np.ndarray,
        sums2: np.ndarray,
    ) -> np.ndarray:
        """Dispatch evaluation to mode-specific kernel."""
        if self._eval_mode == "optim":
            return l2_cost_optim(starts, ends, sums, sums2)
        if self._eval_mode == "fixed":
            return l2_cost_fixed(starts, ends, sums, sums2, self.mu_)
        raise RuntimeError(f"Unknown evaluation mode: {self._eval_mode}")

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for the L2 cost.

        Returns
        -------
        SkchangeTags
            Tags object with L2 cost configuration.
        """
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.score_type = "cost"
        return tags


class CUSUM(BaseIntervalScorer):
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
        validate_data(
            self,
            X,
            ensure_2d=True,
            dtype=np.float64,
            reset=True,
        )
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
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            dtype=np.float64,
            reset=False,
        )
        return {"sums": col_cumsum(X, init_zero=True)}

    def evaluate(self, precomputed: dict, cuts: ArrayLike) -> np.ndarray:
        """Evaluate CUSUM score at splits within intervals.

        Parameters
        ----------
        precomputed : dict
            Precomputed data from :meth:`precompute`.
        cuts : array-like of shape (n_cuts, 3)
            Interval boundaries and split locations ``[start, split, end)``.

        Returns
        -------
        scores : ndarray of shape (n_cuts, n_features)
            CUSUM scores for each cut and feature.
        """
        check_is_fitted(self)
        sums = precomputed["sums"]

        cuts = check_array(cuts, ensure_2d=True, dtype=np.int64)
        if cuts.shape[1] != 3:
            raise ValueError(f"cuts must have shape (n_cuts, 3), got {cuts.shape}.")

        starts = cuts[:, 0]
        splits = cuts[:, 1]
        ends = cuts[:, 2]

        if np.any(splits <= starts) or np.any(splits >= ends):
            raise ValueError(
                "Each cut must satisfy start < split < end for CUSUM evaluation."
            )

        return cusum_score(starts, splits, ends, sums)

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for CUSUM scorer."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.score_type = "change_score"
        return tags


class ChangeScore(BaseIntervalScorer):
    """Change score wrapper constructed from a cost scorer.

    Computes change score as the cost reduction of allowing a split within an interval:

    ``score = cost(start, end) - cost(start, split) - cost(split, end)``
    """

    def __init__(self, cost: BaseIntervalScorer):
        self.cost = cost

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit wrapped cost scorer."""
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            dtype=np.float64,
            reset=True,
        )
        self.cost_: IntervalScorer = clone(self.cost).fit(X, y)
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute wrapped cost scorer data."""
        check_is_fitted(self, ["cost_"])
        return {"cost_precomputed": self.cost_.precompute(X)}

    def evaluate(self, precomputed: dict, cuts: ArrayLike) -> np.ndarray:
        """Evaluate change score for cuts of shape ``(n_cuts, 3)``."""
        check_is_fitted(self, ["cost_"])
        cuts = check_array(cuts, ensure_2d=True, dtype=np.int64)
        if cuts.shape[1] != 3:
            raise ValueError(f"cuts must have shape (n_cuts, 3), got {cuts.shape}.")

        starts = cuts[:, 0]
        splits = cuts[:, 1]
        ends = cuts[:, 2]
        if np.any(splits <= starts) or np.any(splits >= ends):
            raise ValueError(
                "Each cut must satisfy start < split < end for change-score evaluation."
            )

        cost_precomputed = precomputed["cost_precomputed"]
        left_intervals = cuts[:, [0, 1]]
        right_intervals = cuts[:, [1, 2]]
        full_intervals = cuts[:, [0, 2]]

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
    """

    def __init__(
        self,
        score: BaseIntervalScorer,
        penalty: ArrayLike | float | None = None,
    ):
        self.score = score
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

        self.score_ = check_interval_scorer(
            self.score,
            required_tasks=["change_score", "saving", "local_saving"],
            allow_penalised=False,
            clone=True,
            caller_name=self.__class__.__name__,
            arg_name="score",
        )
        self.score_.fit(X, y)

        score_tags = self.score_.__sklearn_tags__().interval_scorer_tags
        if score_tags.aggregated and self.penalty is not None:
            penalty_arr = np.asarray(self.penalty).reshape(-1)
            if penalty_arr.size > 1:
                raise ValueError(
                    "`penalty` must be scalar for aggregated input scores."
                )

        if self.penalty is None:
            penalty = self.score_.get_default_penalty(X.shape[0], X.shape[1])
        else:
            penalty = self.penalty
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
        check_is_fitted(self, ["score_", "penalty_", "_penalty_mode"])
        score_precomputed = self.score_.precompute(X)
        return {"score_precomputed": score_precomputed}

    def evaluate(self, precomputed: dict, cuts: ArrayLike) -> np.ndarray:
        """Evaluate wrapped scores and apply penalty aggregation."""
        check_is_fitted(self, ["score_", "penalty_", "_penalty_mode"])

        scores = self.score_.evaluate(precomputed["score_precomputed"], cuts)
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
    def min_size(self) -> int:
        """Minimum valid interval size inherited from wrapped scorer."""
        check_is_fitted(self)
        return self.score_.min_size

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for penalised scorer."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.score_type = (
            self.score.__sklearn_tags__().interval_scorer_tags.score_type
        )
        tags.interval_scorer_tags.aggregated = True
        tags.interval_scorer_tags.penalised = True
        return tags
