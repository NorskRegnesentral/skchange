"""Classes to construct other types of interval scorers from costs."""

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import (
    BaseChangeScore,
    BaseIntervalScorer,
    BaseTransientScore,
)
from skchange.new_api.types import ArrayLike, Self
from skchange.new_api.utils import SkchangeTags
from skchange.new_api.utils.validation import (
    check_interval_scorer,
    check_interval_specs,
    skip_validation,
    validate_data,
)


class CostChangeScore(BaseChangeScore):
    """Change scorer constructed from a cost scorer.

    Computes change score as the cost reduction of allowing a split within an interval:

    ``score = cost(start, end) - cost(start, split) - cost(split, end)``
    """

    def __init__(self, cost: BaseIntervalScorer):
        self.cost = cost

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit wrapped cost scorer."""
        X = validate_data(self, X, ensure_2d=True, reset=True)
        check_interval_scorer(
            self.cost,
            ensure_score_type=["cost"],
            caller_name=self.__class__.__name__,
            arg_name="cost",
        )
        self.cost_: BaseIntervalScorer = clone(self.cost).fit(X, y)
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute wrapped cost scorer data."""
        check_is_fitted(self, ["cost_"])
        return self.cost_.precompute(X)

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate change score on interval specifications.

        Parameters
        ----------
        cache : dict
            Cache from precompute().
        interval_specs : array-like of shape (n_interval_specs, 3)
            Interval boundaries and split locations ``[start, split, end)``.

        Returns
        -------
        scores : ndarray of shape (n_interval_specs,) or (n_interval_specs, n_features)
            Change scores for each interval specification.
        """
        check_is_fitted(self, ["cost_"])
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            check_sorted=True,
            caller_name=self.__class__.__name__,
        )

        left_intervals = interval_specs[:, [0, 1]]
        right_intervals = interval_specs[:, [1, 2]]
        full_intervals = interval_specs[:, [0, 2]]

        left_costs = self.cost_.evaluate(cache, left_intervals)
        right_costs = self.cost_.evaluate(cache, right_intervals)
        no_change_costs = self.cost_.evaluate(cache, full_intervals)

        change_scores = no_change_costs - (left_costs + right_costs)
        if change_scores.size == 0:
            return change_scores
        min_score = np.min(change_scores)
        if min_score < -1e-6:
            warnings.warn(
                f"{self.cost.__class__.__name__} produced negative change scores "
                f"(min={min_score:.3g}). The cost may not be subadditive.",
                RuntimeWarning,
                stacklevel=2,
            )
        return np.maximum(change_scores, 0.0)

    @property
    def min_size(self) -> int:
        """Minimum valid interval size inherited from wrapped cost scorer."""
        check_is_fitted(self)
        return self.cost_.min_size

    def get_default_penalty(self) -> float | np.ndarray:
        """Get the default penalty delegated to the wrapped cost scorer.

        Returns
        -------
        float or np.ndarray
            Default penalty value from the wrapped cost scorer.
        """
        check_is_fitted(self)
        return self.cost_.get_default_penalty()

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for change score wrapper."""
        tags = super().__sklearn_tags__()
        cost_tags = self.cost.__sklearn_tags__()
        tags.input_tags = cost_tags.input_tags
        tags.interval_scorer_tags.score_type = "change_score"
        tags.interval_scorer_tags.aggregated = cost_tags.interval_scorer_tags.aggregated
        tags.interval_scorer_tags.penalised = cost_tags.interval_scorer_tags.penalised
        tags.interval_scorer_tags.linear_trend_segment = (
            cost_tags.interval_scorer_tags.linear_trend_segment
        )
        return tags


class CostTransientScore(BaseTransientScore):
    """Transient score constructed from a cost scorer.

    Compares the cost of an inner interval and its surrounding (left + right)
    interval against the cost of the outer interval. Each interval spec is a
    4-column row ``[outer_start, inner_start, inner_end, outer_end)``. The
    transient score is

    ``score = cost(outer) - cost(inner) - cost(surrounding)``,

    where ``surrounding`` is the union of ``[outer_start, inner_start)`` and
    ``[inner_end, outer_end)``. The wrapped cost is fit once on the full
    training data; the surrounding evaluation only re-runs ``precompute`` on
    the concatenated surrounding subset, so any fit-time state (e.g. EDF
    quantile points) is shared across inner, outer, and surrounding terms.

    Transient scores correspond to the *epidemic changepoint* model in the
    statistical literature [1]_: a regime change inside ``[inner_start,
    inner_end)`` that returns to the surrounding baseline. They are the
    natural building block for epidemic changepoint detectors such as
    :class:`CircularBinarySegmentation`, in contrast to
    standard (single-shift) change scores used by detectors like :class:`PELT`.

    Parameters
    ----------
    cost : BaseIntervalScorer
        The cost scorer to wrap. Must have ``score_type='cost'``.

    Notes
    -----
    This wrapper is **slow** compared to a native transient score: the
    surrounding-baseline cost is re-precomputed once per candidate inner
    interval, so evaluation time grows linearly in the number of interval
    specs. For the common L2 case prefer :class:`L2TransientScore`, which
    is fully vectorised and ~100-400x faster on typical CBS workloads. For
    other costs, consider implementing a dedicated transient score subclass.

    The wrapper requires the cost to depend only on the multiset of rows in
    the segment it is evaluated on (i.e. it must be invariant to row order
    and independent of any wider reference set). Costs that violate this
    assumption are not subadditive under the concatenated-surrounding
    baseline used here and are explicitly rejected at ``fit`` time:

    - :class:`RankCost` — ranks are recomputed within each segment, so the
      surrounding rank scale differs from the inner/outer rank scale.
    - :class:`LinearTrendCost` — when used with index-based time steps the
      concatenated surrounding collapses the inner gap, so the fitted line
      sees a denser time grid than it should.
    - :class:`MultivariateTCost` — the per-segment log-likelihood is
      obtained from an iterative non-convex MLE, so inner/outer/surrounding
      fits can land in different local optima.

    For these costs, implement a dedicated native transient score (see
    :class:`L2TransientScore` for the pattern) instead.

    References
    ----------
    .. [1] Kirch, C., Muhsal, B., & Ombao, H. (2015). Detection of changes in
        multivariate time series with application to EEG data. Journal of the
        American Statistical Association, 110(511), 1197-1216.
    """

    # Costs whose value depends on row order or on within-segment
    # quantities that change when the surrounding is rebuilt by
    # concatenation. Listed by class name to avoid import cycles.
    _INCOMPATIBLE_COST_NAMES = frozenset(
        {"RankCost", "LinearTrendCost", "MultivariateTCost"}
    )

    def __init__(self, cost: BaseIntervalScorer):
        self.cost = cost

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit wrapped cost scorer."""
        X = validate_data(self, X, ensure_2d=True, reset=True)
        check_interval_scorer(
            self.cost,
            ensure_score_type=["cost"],
            caller_name=self.__class__.__name__,
            arg_name="cost",
        )
        cost_name = type(self.cost).__name__
        if cost_name in self._INCOMPATIBLE_COST_NAMES:
            raise ValueError(
                f"{cost_name} is not compatible with CostTransientScore: its "
                "value depends on row order or on a within-segment reference "
                "that changes when the surrounding interval is rebuilt by "
                "concatenation, so the resulting transient score is not "
                "subadditive. Implement a dedicated native transient score "
                "(see L2TransientScore for the pattern) instead."
            )
        self.cost_: BaseIntervalScorer = clone(self.cost).fit(X, y)
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute wrapped cost data and store ``X`` for surrounding refits."""
        check_is_fitted(self, ["cost_"])
        X = np.asarray(X)
        cache = self.cost_.precompute(X)
        # Wrap in a new dict so we don't mutate the wrapped cost's cache.
        return {"cost_cache": cache, "X": X}

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate transient score on interval specifications.

        Parameters
        ----------
        cache : dict
            Cache from ``precompute()``.
        interval_specs : array-like of shape (n_interval_specs, 4)
            Interval boundaries
            ``[outer_start, inner_start, inner_end, outer_end)``.

        Returns
        -------
        scores : ndarray of shape (n_interval_specs,) or (n_interval_specs, n_features)
            Transient scores for each interval specification.
        """
        check_is_fitted(self, ["cost_"])
        # Use ``check_sorted=False``: a transient interval ``[outer_s, inner_s,
        # inner_e, outer_e)`` only requires ``outer_s <= inner_s < inner_e <=
        # outer_e``. The inner anomaly interval must be non-empty, but the
        # surrounding "before" or "after" segment is allowed to be empty (the
        # surrounding-cost loop below concatenates them, so ``np.concatenate``
        # naturally handles the empty case).
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            check_sorted=False,
            caller_name=self.__class__.__name__,
        )
        if interval_specs.size > 0:
            outer_s = interval_specs[:, 0]
            inner_s = interval_specs[:, 1]
            inner_e = interval_specs[:, 2]
            outer_e = interval_specs[:, 3]
            valid = (outer_s <= inner_s) & (inner_s < inner_e) & (inner_e <= outer_e)
            if not np.all(valid):
                raise ValueError(
                    f"Each row of `interval_specs` must satisfy "
                    f"outer_start <= inner_start < inner_end <= outer_end "
                    f"in {self.__class__.__name__}."
                )

        cost_cache = cache["cost_cache"]
        X = cache["X"]

        inner_intervals = interval_specs[:, [1, 2]]
        outer_intervals = interval_specs[:, [0, 3]]
        inner_costs = self.cost_.evaluate(cost_cache, inner_intervals)
        outer_costs = self.cost_.evaluate(cost_cache, outer_intervals)

        surrounding_costs = np.zeros_like(outer_costs)
        cost_ = self.cost_
        # The surrounding cost is recomputed once per inner candidate, so the
        # per-call sklearn validation overhead dominates. ``X`` and the
        # ``[[0, n]]`` spec are trivially well-formed, so we bypass all input
        # checks inside the wrapped cost's ``precompute``/``evaluate`` via
        # ``skip_validation()``.
        surrounding_spec = np.zeros((1, 2), dtype=np.intp)
        with skip_validation():
            for i in range(interval_specs.shape[0]):
                outer_s = interval_specs[i, 0]
                inner_s = interval_specs[i, 1]
                inner_e = interval_specs[i, 2]
                outer_e = interval_specs[i, 3]
                before_data = X[outer_s:inner_s]
                after_data = X[inner_e:outer_e]
                surrounding_data = np.concatenate((before_data, after_data))
                # Reuse the wrapped cost fitted on the full training data;
                # only ``precompute`` is re-run on the (contiguous)
                # surrounding subset.
                surrounding_cache = cost_.precompute(surrounding_data)
                surrounding_spec[0, 1] = surrounding_data.shape[0]
                surrounding_costs[i] = cost_.evaluate(
                    surrounding_cache, surrounding_spec
                )[0]

        transient_scores = outer_costs - (inner_costs + surrounding_costs)
        if transient_scores.size == 0:
            return transient_scores
        min_score = np.min(transient_scores)
        if min_score < -1e-6:
            warnings.warn(
                f"{self.cost.__class__.__name__} produced negative transient scores "
                f"(min={min_score:.3g}). The cost may not be subadditive.",
                RuntimeWarning,
                stacklevel=2,
            )
        return np.maximum(transient_scores, 0.0)

    @property
    def min_size(self) -> int:
        """Minimum valid inner-interval size inherited from wrapped cost scorer.

        Note that the surrounding (left + right) interval must also have at least
        ``min_size`` total samples, so the outer interval must be at least
        ``2 * min_size`` long.
        """
        check_is_fitted(self)
        return self.cost_.min_size

    def get_default_penalty(self) -> float | np.ndarray:
        """Get the default penalty delegated to the wrapped cost scorer."""
        check_is_fitted(self)
        return self.cost_.get_default_penalty()

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for transient score wrapper."""
        tags = super().__sklearn_tags__()
        cost_tags = self.cost.__sklearn_tags__()
        tags.input_tags = cost_tags.input_tags
        tags.interval_scorer_tags.score_type = "transient_score"
        tags.interval_scorer_tags.aggregated = cost_tags.interval_scorer_tags.aggregated
        tags.interval_scorer_tags.penalised = cost_tags.interval_scorer_tags.penalised
        tags.interval_scorer_tags.linear_trend_segment = (
            cost_tags.interval_scorer_tags.linear_trend_segment
        )
        return tags
