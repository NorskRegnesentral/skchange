"""Seeded binary segmentation algorithm for multiple changepoint detection."""

__author__ = ["Tveten"]
__all__ = ["SeededBinarySegmentation"]

from numbers import Integral, Real

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.detectors._base import BaseChangeDetector
from skchange.new_api.interval_scorers._base import BaseIntervalScorer
from skchange.new_api.interval_scorers._change_scores.cusum import CUSUM
from skchange.new_api.types import ArrayLike, Self
from skchange.new_api.utils import SkchangeTags
from skchange.new_api.utils._numba import njit
from skchange.new_api.utils._param_validation import (
    HasMethods,
    Interval,
    StrOptions,
    _fit_context,
)
from skchange.new_api.utils._score_aggregation import (
    USER_AGG_CHOICES,
    aggregate_and_penalise,
    resolve_aggregation,
    resolve_penalty,
)
from skchange.new_api.utils.validation import (
    check_interval_scorer,
    validate_data,
)


@njit(cache=True)
def make_seeded_intervals(
    n: int, min_length: int, max_length: int, growth_factor: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    starts = [0]  # For numba to be able to compile type.
    ends = [1]  # For numba to be able to compile type.
    step_factor = 1 - 1 / growth_factor
    max_length = min(max_length, n)
    if max_length < min_length:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    n_lengths = max(
        1, int(np.ceil(np.log(max_length / min_length) / np.log(growth_factor)))
    )
    interval_lens = np.unique(np.round(np.geomspace(min_length, max_length, n_lengths)))
    for interval_len in interval_lens:
        step = max(1, np.round(step_factor * interval_len))
        n_steps = int(np.ceil((n - interval_len) / step))
        new_starts = [int(i * step) for i in range(n_steps + 1)]
        starts += new_starts
        new_ends = [int(min(i * step + interval_len, n)) for i in range(n_steps + 1)]
        ends += new_ends
        if ends[-1] - starts[-1] < min_length:
            starts[-1] = n - min_length
    return np.array(starts[1:]), np.array(ends[1:])


@njit(cache=True)
def greedy_selection(
    max_scores: np.ndarray,
    argmax_scores: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> list[int]:
    max_scores = max_scores.copy()
    cpts = []
    while np.any(max_scores > 0):
        argmax = max_scores.argmax()
        cpt = argmax_scores[argmax]
        cpts.append(int(cpt))
        # remove intervals that contain the detected changepoint.
        max_scores[(cpt >= starts) & (cpt < ends)] = 0.0
    cpts.sort()
    return cpts


@njit(cache=True)
def narrowest_selection(
    max_scores: np.ndarray,
    argmax_scores: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> list[int]:
    cpts = []
    scores_above_threshold = max_scores > 0
    candidate_starts = starts[scores_above_threshold]
    candidate_ends = ends[scores_above_threshold]
    candidate_maximizers = argmax_scores[scores_above_threshold]

    while len(candidate_starts) > 0:
        argmin = np.argmin(candidate_ends - candidate_starts)
        cpt = candidate_maximizers[argmin]
        cpts.append(int(cpt))

        # remove candidates that contain the detected changepoint.
        cpt_not_in_interval = ~((cpt >= candidate_starts) & (cpt < candidate_ends))
        candidate_starts = candidate_starts[cpt_not_in_interval]
        candidate_ends = candidate_ends[cpt_not_in_interval]
        candidate_maximizers = candidate_maximizers[cpt_not_in_interval]

    cpts.sort()
    return cpts


def _score_seeded_intervals(
    change_score: BaseIntervalScorer,
    agg_mode: str,
    penalty: float | np.ndarray | None,
    X: np.ndarray,
    min_subinterval_length: int,
    max_interval_length: int,
    growth_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score the seeded interval grid without changepoint selection.

    Parameters
    ----------
    change_score : BaseIntervalScorer
        Fitted scorer used to evaluate change scores on the seeded intervals.
        ``change_score.precompute(X)`` and ``change_score.evaluate(cache, specs)``
        are called. May be unpenalised (per-feature scores) or inherently
        penalised (already-aggregated single-column scores).
    agg_mode : str
        Aggregation/penalty mode as returned by ``resolve_aggregation``.
    penalty : float, np.ndarray or None
        Effective penalty as returned by ``resolve_aggregation``. ``None``
        when ``agg_mode == "passthrough"``.
    X : np.ndarray of shape (n_samples, n_features)
        Input data.
    min_subinterval_length : int
        Minimum length of a subinterval on each side of a candidate split. Must
        be at least ``change_score.min_size``.
    max_interval_length : int
        Maximum length of an interval to evaluate.
    growth_factor : float
        Growth factor for the seeded intervals.

    Returns
    -------
    max_scores : np.ndarray
        Maximum (aggregated, penalised) score per seeded interval.
    argmax_scores : np.ndarray
        Index of the maximum-score split for each seeded interval.
    starts : np.ndarray
        Start indices of the seeded intervals.
    ends : np.ndarray
        End indices of the seeded intervals.
    """
    check_is_fitted(change_score)
    cache = change_score.precompute(X)
    starts, ends = make_seeded_intervals(
        X.shape[0],
        2 * min_subinterval_length,
        max_interval_length,
        growth_factor,
    )

    max_scores = np.zeros(starts.size)
    argmax_scores = np.zeros(starts.size, dtype=np.int64)

    if starts.size > 0:
        splits_per_interval = [
            np.arange(s + min_subinterval_length, e - min_subinterval_length + 1)
            for s, e in zip(starts, ends)
        ]
        n_splits = np.fromiter(
            (sp.size for sp in splits_per_interval),
            dtype=np.intp,
            count=starts.size,
        )
        all_splits = np.concatenate(splits_per_interval)
        all_starts = np.repeat(starts, n_splits)
        all_ends = np.repeat(ends, n_splits)
        interval_specs = np.column_stack((all_starts, all_splits, all_ends))

        # Evaluate the change score on all specs in a single call. This is
        # much faster than calling ``change_score.evaluate`` once per interval.
        raw_scores = change_score.evaluate(cache, interval_specs)
        penalised_scores = aggregate_and_penalise(raw_scores, agg_mode, penalty)

        # Split the flat score array back per interval to find the per-interval
        # max and argmax. The loop is over the (small) number of intervals only.
        offsets = np.concatenate(([0], np.cumsum(n_splits)))
        for i in range(starts.size):
            interval_scores = penalised_scores[offsets[i] : offsets[i + 1]]
            argmax = int(np.argmax(interval_scores))
            max_scores[i] = interval_scores[argmax]
            argmax_scores[i] = splits_per_interval[i][argmax]

    return max_scores, argmax_scores, starts, ends


def _select_changepoints(
    max_scores: np.ndarray,
    argmax_scores: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    selection_method: str,
) -> np.ndarray:
    """Greedy or narrowest selection of changepoints from per-interval scores."""
    if selection_method == "greedy":
        cpts = greedy_selection(max_scores, argmax_scores, starts, ends)
    else:  # "narrowest"
        cpts = narrowest_selection(max_scores, argmax_scores, starts, ends)
    return np.array(cpts, dtype=np.intp)


def _resolve_change_score(
    change_score: BaseIntervalScorer | None,
) -> BaseIntervalScorer:
    """Resolve default change score.

    Needed since default resolution must be done in both fit and
    ``__sklearn_tags__`` to ensure correct input tags are propagated.
    """
    return CUSUM() if change_score is None else change_score


class SeededBinarySegmentation(BaseChangeDetector):
    """Seeded binary segmentation algorithm for multiple changepoint detection.

    Binary segmentation type changepoint detection algorithms recursively split the data
    into two segments and test whether the two segments are different. The seeded binary
    segmentation algorithm is an efficient version of such algorithms that tests for
    changepoints in intervals of exponentially growing length. It has the same
    theoretical guarantees as the original binary segmentation algorithm but runs in
    log-linear time regardless of the changepoint configuration.

    The ``penalty``, ``penalty_scale``, and ``agg`` parameters are ignored
    when ``change_score`` is an inherently penalised scorer (tag
    ``penalised=True``); in that case the scorer owns aggregation and
    penalisation.

    Parameters
    ----------
    change_score : BaseIntervalScorer or None, default=None
        Change score to use in the algorithm. Must be an instance of
        ``BaseIntervalScorer`` with ``score_type="change_score"``. If ``None``,
        defaults to ``CUSUM()``.

        Standard usage is to pass an unpenalised score and set ``penalty``
        separately. If the score already has tag ``penalised=True``, it owns
        its own aggregation and penalty, and ``penalty`` / ``penalty_scale``
        are ignored.
    penalty : float, array-like of shape (n_features,) or None, default=None
        Penalty subtracted from the aggregated feature-wise score. A candidate
        changepoint is accepted only when the penalised score is positive.

        - ``float``: scalar penalty subtracted from the aggregated score
          (see ``agg``).
        - ``array-like`` of length ``n_features``, non-decreasing: element
          ``i`` is the penalty for ``i+1`` features jointly affected; the
          detector picks the ``k`` largest feature scores maximising
          ``sum(top_k) - penalty[k-1]`` (handles sparse changes). A strictly
          linear array uses a faster code path. Only consistent with the
          default ``agg="sum"``.
        - ``None``: defaults to ``change_score.get_default_penalty()``
          (BIC-based) at fit time.

        Ignored when ``change_score`` is already penalised.
    penalty_scale : float, default=1.0
        Multiplicative factor applied to the effective penalty (whether
        ``penalty`` is user-provided or the default). Use as a single scalar tuning knob
        that preserves the shape of array penalties. Must be positive. Ignored when
        ``change_score`` is already penalised.
    agg : {"sum", "max"}, default="sum"
        How feature-wise raw scores are aggregated into a single score per
        interval before subtracting a scalar penalty:

        * ``"sum"``: ``sum(scores, axis=1) - penalty`` (dense change
          assumption).
        * ``"max"``: ``max(scores, axis=1) - penalty`` (a single, unknown
          feature changes).

        Ignored when ``change_score`` is already aggregated. Must be
        ``"sum"`` when ``penalty`` is array-valued (array penalties imply
        top-k aggregation).
    min_subinterval_length : int, default=5
        Minimum length of a subinterval on each side of a candidate split point
        within each evaluated interval. The effective minimum used is
        ``max(min_subinterval_length, change_score.min_size)``, so the actual
        minimum may be larger than the value provided here when the change
        score requires more samples per segment. Note that this does not impose
        a lower bound on the spacing between detected changepoints.
    max_interval_length : int or None, default=None
        The maximum length of an interval to evaluate a changepoint in. Must be at
        least ``2 * min_subinterval_length``. If ``None``, defaults to
        ``min(200, n_samples)`` after fitting.
    growth_factor : float, default=1.5
        Growth factor for the seeded intervals. Intervals grow in size according to
        ``interval_len = max(interval_len + 1, floor(growth_factor * interval_len))``,
        starting at ``interval_len = 2 * min_subinterval_length``. It also governs the
        amount of overlap between intervals of the same length, as the start of each
        interval is shifted by a factor of ``1 - 1 / growth_factor``. Larger values
        produce fewer, less-overlapping intervals (faster but coarser); smaller values
        produce more, more-overlapping intervals (slower but finer).
        Must be in ``(1, 2]``.
    selection_method : str, default="greedy"
        Method for selecting the final set of changepoints from candidate intervals
        with positive penalised score. Options:

        * ``"greedy"``: Select the interval with the highest score, remove all
          overlapping intervals containing the detected changepoint, and repeat
          until no intervals remain with a positive score.
        * ``"narrowest"``: Among intervals with positive scores, select the
          narrowest one, remove all overlapping intervals containing the detected
          changepoint, and repeat. Corresponds to the narrowest-over-threshold
          approach of [2]_.

    Attributes
    ----------
    change_score_ : BaseIntervalScorer
        Fitted change score. When ``change_score`` is unpenalised this is the
        fitted unpenalised scorer (same type as the input). When the input is
        already penalised, it is that fitted penalised scorer.
    penalty_ : float, np.ndarray or None
        Effective penalty actually used at detection time: the resolved base
        penalty (either ``penalty`` or, if ``None``,
        ``change_score.get_default_penalty()``) multiplied by
        ``penalty_scale``. ``None`` when ``change_score`` is inherently
        penalised, in which case the scorer owns its own penalty.
    min_subinterval_length_ : int
        Effective minimum split size used.
    max_interval_length_ : int
        Effective maximum interval length used.

    References
    ----------
    .. [1] Kovács, S., Bühlmann, P., Li, H., & Munk, A. (2023). Seeded binary
        segmentation: a general methodology for fast and optimal changepoint detection.
        Biometrika, 110(1), 249-256.

    .. [2] Baranowski, R., Chen, Y., & Fryzlewicz, P. (2019). Narrowest-over-threshold
        detection of multiple change points and change-point-like features. Journal of
        the Royal Statistical Society Series B: Statistical Methodology, 81(3), 649-672.

    Notes
    -----
    Typical usage recipes:

    * Default::

        SeededBinarySegmentation()

    * Tune sensitivity on a log grid::

        SeededBinarySegmentation(penalty_scale=5.0)

    * Detect changes in a single, unknown feature (sparse case)::

        SeededBinarySegmentation(agg="max")

    * Adaptive sparsity via an array penalty (top-k aggregation)::

        SeededBinarySegmentation(penalty=linear_chi2_penalty(n, n_features))

    * Power-user path with a joint scorer that owns aggregation and penalty::

        SeededBinarySegmentation(change_score=ESACScore(...))

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.detectors import SeededBinarySegmentation
    >>> rng = np.random.default_rng(2)
    >>> X = np.concatenate([rng.normal(0, 1, (100, 1)),
    ...                     rng.normal(10, 1, (100, 1))])
    >>> detector = SeededBinarySegmentation()
    >>> detector.fit_predict(X)
    array([100])
    """

    _calibration_strategy = "max_score"

    _parameter_constraints = {
        "change_score": [HasMethods(["fit", "precompute", "evaluate"]), None],
        "penalty": ["array-like", Interval(Real, 0, None, closed="left"), None],
        "penalty_scale": [Interval(Real, 0, None, closed="neither")],
        "agg": [StrOptions(set(USER_AGG_CHOICES))],
        "min_subinterval_length": [Interval(Integral, 1, None, closed="left")],
        "max_interval_length": [Interval(Integral, 2, None, closed="left"), None],
        "growth_factor": [Interval(Real, 1.0, 2.0, closed="right")],
        "selection_method": [StrOptions({"greedy", "narrowest"})],
    }

    def __init__(
        self,
        change_score: BaseIntervalScorer | None = None,
        penalty: ArrayLike | float | None = None,
        penalty_scale: float = 1.0,
        agg: str = "sum",
        min_subinterval_length: int = 5,
        max_interval_length: int | None = None,
        growth_factor: float = 1.5,
        selection_method: str = "greedy",
    ):
        self.change_score = change_score
        self.penalty = penalty
        self.penalty_scale = penalty_scale
        self.agg = agg
        self.min_subinterval_length = min_subinterval_length
        self.max_interval_length = max_interval_length
        self.growth_factor = growth_factor
        self.selection_method = selection_method

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get tags, propagating input constraints from the wrapped scorer."""
        tags = super().__sklearn_tags__()
        scorer_tags = _resolve_change_score(self.change_score).__sklearn_tags__()
        tags.input_tags = scorer_tags.input_tags
        tags.change_detector_tags.linear_trend_segment = (
            scorer_tags.interval_scorer_tags.linear_trend_segment
        )
        return tags

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the change score to training data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training time series data.
        y : ArrayLike | None, default=None
            Ignored.

        Returns
        -------
        self : SeededBinarySegmentation
            Fitted detector.
        """
        X = validate_data(self, X, reset=True, ensure_2d=True)

        change_score = _resolve_change_score(self.change_score)
        check_interval_scorer(
            change_score,
            ensure_score_type=["change_score"],
            caller_name=self.__class__.__name__,
            arg_name="change_score",
        )
        self.change_score_ = clone(change_score).fit(X, y)

        self.penalty_ = resolve_penalty(
            self.change_score_,
            self.penalty,
            self.penalty_scale,
            caller_name=self.__class__.__name__,
        )
        self._agg_mode = resolve_aggregation(
            self.change_score_,
            self.agg,
            self.penalty_,
            self.n_features_in_,
            caller_name=self.__class__.__name__,
        )

        self.min_subinterval_length_ = max(
            self.min_subinterval_length, self.change_score_.min_size
        )
        if self.n_samples_in_ < 2 * self.min_subinterval_length_:
            raise ValueError(
                f"`SeededBinarySegmentation` requires at least "
                f"2 * min_subinterval_length (={2 * self.min_subinterval_length_}) "
                f"samples to fit, got "
                f"n_samples={self.n_samples_in_}."
            )

        if self.max_interval_length is None:
            self.max_interval_length_ = min(200, self.n_samples_in_)
        else:
            self.max_interval_length_ = self.max_interval_length

        if self.max_interval_length_ < 2 * self.min_subinterval_length_:
            raise ValueError(
                f"`max_interval_length` (={self.max_interval_length_}) must be at "
                f"least 2 * min_subinterval_length "
                f"(={2 * self.min_subinterval_length_})."
            )

        return self

    def predict_all(self, X: ArrayLike) -> dict:
        """Run seeded binary segmentation and return all outputs in a single pass.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyse for changepoints.

        Returns
        -------
        result : dict with keys:

            ``"changepoints"`` : np.ndarray of shape (n_changepoints,)
                Sorted integer indices of detected changepoints.
            ``"interval_starts"`` : np.ndarray
                Start indices of the seeded intervals evaluated.
            ``"interval_ends"`` : np.ndarray
                End indices of the seeded intervals evaluated.
            ``"interval_max_scores"`` : np.ndarray
                Maximum penalised score within each seeded interval.
            ``"interval_argmax_splits"`` : np.ndarray
                Index of the best split (changepoint candidate) per interval.
        """
        check_is_fitted(self)

        max_scores, index = self.predict_scores(X, return_index=True)
        starts = index["starts"]
        ends = index["ends"]
        argmax_scores = index["argmax_splits"]
        cpts = _select_changepoints(
            max_scores, argmax_scores, starts, ends, self.selection_method
        )
        return {
            "changepoints": cpts,
            "interval_starts": starts,
            "interval_ends": ends,
            "interval_max_scores": max_scores,
            "interval_argmax_splits": argmax_scores,
        }

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Detect changepoints in a time series.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyse for changepoints.

        Returns
        -------
        changepoints : np.ndarray of shape (n_changepoints,)
            Sorted integer indices of detected changepoints. A changepoint is defined
            as the first index of a segment, such that the data segments are given
            by ``X[:cpt[0]], X[cpt[0]:cpt[1]], ..., X[cpt[-1]:]``.
            Empty array if no changepoints are detected.
        """
        return self.predict_all(X)["changepoints"]

    def predict_scores(
        self,
        X: ArrayLike,
        return_index: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return the per-interval scoring objective evaluated on ``X``.

        Returns the per-interval maximum of the aggregated, penalised change
        score across candidate splits, using the exact same seeded intervals
        and candidate splits as :meth:`predict`. The output is what
        :meth:`predict` reduces over via the selection step, without the
        selection itself.

        For penalty calibration, use the free function
        :func:`skchange.new_api.tuning.unpenalised_scores`, which fits a clone
        of this detector with the penalty parameter set to zero and returns
        the resulting unpenalised scores.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to evaluate.
        return_index : bool, default=False
            If ``True``, also return a dict locating each score on the time
            axis. See the Returns section for the keys.

        Returns
        -------
        scores : np.ndarray of shape (n_intervals,)
            Aggregated, penalised change-score values, one per seeded
            interval, at the best split within that interval. Returned alone
            when ``return_index=False``.
        index : dict, optional
            Only returned when ``return_index=True``. Contains:

            - ``"starts"`` : np.ndarray of shape (n_intervals,)
              Start indices of the seeded intervals.
            - ``"ends"`` : np.ndarray of shape (n_intervals,)
              End indices of the seeded intervals.
            - ``"argmax_splits"`` : np.ndarray of shape (n_intervals,)
              Index of the maximising split within each interval.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True)

        max_scores, argmax_splits, starts, ends = _score_seeded_intervals(
            change_score=self.change_score_,
            agg_mode=self._agg_mode,
            penalty=self.penalty_,
            X=X,
            min_subinterval_length=self.min_subinterval_length_,
            max_interval_length=self.max_interval_length_,
            growth_factor=self.growth_factor,
        )

        if return_index:
            return max_scores, {
                "starts": starts,
                "ends": ends,
                "argmax_splits": argmax_splits,
            }
        return max_scores
