"""Circular binary segmentation algorithm for multiple transient change detection."""

__author__ = ["Tveten"]
__all__ = ["CircularBinarySegmentation"]

from numbers import Integral, Real

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.detectors._base import BaseChangeDetector
from skchange.new_api.detectors._seeded_binseg import make_seeded_intervals
from skchange.new_api.interval_scorers._base import BaseIntervalScorer
from skchange.new_api.interval_scorers._transient_scores.l2_transient_score import (
    L2TransientScore,
)
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
def make_inner_intervals(
    interval_start: int, interval_end: int, min_subinterval_length: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Generate inner-interval candidates within an outer interval.

    For an outer interval ``[interval_start, interval_end)``, returns all
    ``(inner_start, inner_end)`` pairs such that:

    - ``inner_end - inner_start >= min_subinterval_length`` (inner segment),
    - ``(inner_start - interval_start) + (interval_end - inner_end)
      >= min_subinterval_length`` (combined surrounding baseline),
    - the inner interval lies inside the outer interval (boundary-touching
      allowed, see below).
    """
    starts = []
    ends = []
    # ``i`` and ``j`` may equal the outer interval boundaries: the inner
    # interval is allowed to abut the start or end of the outer interval, in
    # which case the corresponding "before" or "after" surrounding segment is
    # empty. The ``baseline_n >= min_subinterval_length`` check ensures the
    # combined surrounding (left + right) is still long enough to fit the cost.
    for i in range(interval_start, interval_end - min_subinterval_length + 2):
        for j in range(i + min_subinterval_length, interval_end + 1):
            baseline_n = interval_end - j + i - interval_start
            if baseline_n >= min_subinterval_length:
                starts.append(i)
                ends.append(j)
    return np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


@njit(cache=True)
def greedy_segment_selection(
    penalised_scores: np.ndarray,
    inner_starts: np.ndarray,
    inner_ends: np.ndarray,
    outer_starts: np.ndarray,
    outer_ends: np.ndarray,
) -> np.ndarray:
    """Greedily select non-overlapping segments with positive score."""
    penalised_scores = penalised_scores.copy()
    segments = []
    while np.any(penalised_scores > 0):
        argmax = penalised_scores.argmax()
        segment_start = inner_starts[argmax]
        segment_end = inner_ends[argmax]
        segments.append((segment_start, segment_end))
        # Remove outer intervals that overlap with the detected segment.
        penalised_scores[
            (segment_end > outer_starts) & (segment_start < outer_ends)
        ] = 0.0

    if len(segments) == 0:
        return np.empty((0, 2), dtype=np.intp)
    segments.sort()
    return np.array(segments, dtype=np.intp)


def _score_circular_intervals(
    transient_score: BaseIntervalScorer,
    agg_mode: str,
    penalty: float | np.ndarray | None,
    X: np.ndarray,
    min_subinterval_length: int,
    max_interval_length: int,
    growth_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score the seeded outer-interval grid without segment selection.

    Parameters
    ----------
    transient_score : BaseIntervalScorer
        Fitted transient score. May be unpenalised (per-feature scores) or
        inherently penalised (already-aggregated single-column scores).
    agg_mode : str
        Aggregation/penalty mode as returned by :func:`resolve_aggregation`.
    penalty : float, np.ndarray or None
        Effective penalty as returned by :func:`resolve_penalty`. ``None``
        when ``agg_mode == "passthrough"``.
    X : np.ndarray of shape (n_samples, n_features)
        Input data.
    min_subinterval_length : int
        Minimum length of an inner interval, and minimum total length of
        the surrounding (left + right) baseline. Must be at least
        ``transient_score.min_size``.
    max_interval_length : int
        Maximum length of an outer interval to evaluate.
    growth_factor : float
        Growth factor for the seeded intervals.

    Returns
    -------
    max_scores : np.ndarray of shape (n_outer,)
        Best (aggregated, penalised) score per outer interval.
    argmax_inner_starts : np.ndarray of shape (n_outer,)
        Inner-interval start of the best inner candidate per outer interval.
    argmax_inner_ends : np.ndarray of shape (n_outer,)
        Inner-interval end of the best inner candidate per outer interval.
    starts : np.ndarray of shape (n_outer,)
        Outer-interval start indices.
    ends : np.ndarray of shape (n_outer,)
        Outer-interval end indices.
    """
    check_is_fitted(transient_score)
    cache = transient_score.precompute(X)
    n_samples = X.shape[0]

    starts, ends = make_seeded_intervals(
        n_samples,
        2 * min_subinterval_length,
        max_interval_length,
        growth_factor,
    )

    max_scores = np.zeros(starts.size)
    argmax_inner_starts = np.zeros(starts.size, dtype=np.int64)
    argmax_inner_ends = np.zeros(starts.size, dtype=np.int64)

    if starts.size == 0:
        return max_scores, argmax_inner_starts, argmax_inner_ends, starts, ends

    # Build the (outer_start, inner_start, inner_end, outer_end) specs for all
    # inner candidates across every outer interval and evaluate the transient
    # score in a single call. Same approach as in ``_score_seeded_intervals``.
    inner_per_interval = [
        make_inner_intervals(start, end, min_subinterval_length)
        for start, end in zip(starts, ends)
    ]
    n_inner = np.fromiter(
        (inner_starts.size for inner_starts, _ in inner_per_interval),
        dtype=np.intp,
        count=starts.size,
    )
    all_inner_starts = np.concatenate(
        [inner_starts for inner_starts, _ in inner_per_interval]
    )
    all_inner_ends = np.concatenate(
        [inner_ends for _, inner_ends in inner_per_interval]
    )
    all_outer_starts = np.repeat(starts, n_inner)
    all_outer_ends = np.repeat(ends, n_inner)
    interval_specs = np.column_stack(
        (all_outer_starts, all_inner_starts, all_inner_ends, all_outer_ends)
    )
    raw_scores = transient_score.evaluate(cache, interval_specs)
    penalised_scores = aggregate_and_penalise(raw_scores, agg_mode, penalty)

    offsets = np.concatenate(([0], np.cumsum(n_inner)))
    for i in range(starts.size):
        interval_scores = penalised_scores[offsets[i] : offsets[i + 1]]
        argmax = int(np.argmax(interval_scores))
        max_scores[i] = interval_scores[argmax]
        inner_starts_i, inner_ends_i = inner_per_interval[i]
        argmax_inner_starts[i] = inner_starts_i[argmax]
        argmax_inner_ends[i] = inner_ends_i[argmax]

    return max_scores, argmax_inner_starts, argmax_inner_ends, starts, ends


def _resolve_transient_score(
    transient_score: BaseIntervalScorer | None,
) -> BaseIntervalScorer:
    """Resolve default transient score.

    Needed since default resolution must be done in both fit and
    ``__sklearn_tags__`` to ensure correct input tags are propagated.
    """
    return L2TransientScore() if transient_score is None else transient_score


class CircularBinarySegmentation(BaseChangeDetector):
    """Circular binary segmentation algorithm for multiple segment anomaly detection.

    Binary segmentation type changepoint detection algorithms recursively split the data
    into two segments and test whether the two segments are different. Circular binary
    segmentation [1]_ is a variant of binary segmentation where the statistical test
    (transient score) compares the data behaviour of an inner interval subset with the
    surrounding data contained in an outer interval. In other words, the null
    hypothesis within each outer interval is that the data is stationary, while the
    alternative hypothesis is that there is a segment anomaly within the outer
    interval.

    Each detected segment anomaly ``[start, end)`` corresponds to a pair of
    *epidemic changepoints* in the statistical literature [2]_: the regime
    transitions in (at ``start``) and out (at ``end``) of a transient segment
    that returns to the surrounding baseline. CBS is therefore an epidemic
    changepoint detector, in contrast to standard (single-shift) changepoint
    methods such as :class:`PELT` or :class:`SeededBinarySegmentation`.

    The ``penalty``, ``penalty_scale``, and ``agg`` parameters are ignored
    when ``transient_score`` is an inherently penalised scorer (tag
    ``penalised=True``); in that case the scorer owns aggregation and
    penalisation.

    Parameters
    ----------
    transient_score : BaseIntervalScorer or None, default=None
        Transient score to use in the algorithm. Must be an instance of
        ``BaseIntervalScorer`` with ``score_type="transient_score"``. If
        ``None``, defaults to ``L2TransientScore()``.

        Standard usage is to pass an unpenalised score and set ``penalty``
        separately. If the score already has tag ``penalised=True``, it owns
        its own aggregation and penalty, and ``penalty`` / ``penalty_scale``
        / ``agg`` are ignored.
    penalty : float, array-like of shape (n_features,) or None, default=None
        Penalty subtracted from the aggregated feature-wise score. A candidate
        segment anomaly is accepted only when the penalised score is positive.

        - ``float``: scalar penalty subtracted from the aggregated score
          (see ``agg``).
        - ``array-like`` of length ``n_features``, non-decreasing: element
          ``i`` is the penalty for ``i+1`` features jointly affected; the
          detector picks the ``k`` largest feature scores maximising
          ``sum(top_k) - penalty[k-1]``. Only consistent with the default
          ``agg="sum"``.
        - ``None``: defaults to ``transient_score.get_default_penalty()`` at
          fit time.

        Ignored when ``transient_score`` is already penalised.
    penalty_scale : float, default=2.0
        Multiplicative factor applied to the effective penalty (whether
        ``penalty`` is user-provided or the default). Must be positive.
        Ignored when ``transient_score`` is already penalised. The default
        is larger than 1 because CBS evaluates the score over a very large
        number of candidate ``(outer, inner)`` interval pairs, so a stricter
        penalty is needed to keep the family-wise false-positive rate low.
    agg : {"sum", "max"}, default="sum"
        How feature-wise raw scores are aggregated into a single score per
        interval before subtracting a scalar penalty:

        * ``"sum"``: ``sum(scores, axis=1) - penalty`` (dense change
          assumption).
        * ``"max"``: ``max(scores, axis=1) - penalty`` (a single, unknown
          feature changes).

        Ignored when ``transient_score`` is already aggregated. Must be
        ``"sum"`` when ``penalty`` is array-valued (array penalties imply
        top-k aggregation).
    min_subinterval_length : int, default=5
        Minimum length of an inner (anomalous) segment. The total length of the
        surrounding (left + right) baseline must also be at least this value.
        The effective minimum used is
        ``max(min_subinterval_length, transient_score.min_size)``.
    max_interval_length : int or None, default=None
        Maximum length of an outer interval to evaluate. Must be at least
        ``2 * min_subinterval_length``. If ``None``, defaults to
        ``min(200, n_samples)`` after fitting.
    growth_factor : float, default=1.8
        Growth factor for the seeded outer intervals. Larger values produce
        fewer, less-overlapping intervals (faster but coarser); smaller values
        produce more, more-overlapping intervals (slower but finer). Must be in
        ``(1, 2]``.

    Attributes
    ----------
    transient_score_ : BaseIntervalScorer
        Fitted transient score.
    penalty_ : float, np.ndarray or None
        Effective penalty actually used at detection time. ``None`` when
        ``transient_score`` is inherently penalised.
    min_subinterval_length_ : int
        Effective minimum inner-segment length used.
    max_interval_length_ : int
        Effective maximum outer-interval length used.

    References
    ----------
    .. [1] Olshen, A. B., Venkatraman, E. S., Lucito, R., & Wigler, M. (2004).
        Circular binary segmentation for the analysis of array-based DNA copy
        number data. Biostatistics, 5(4), 557-572.
    .. [2] Levin, B. & Kline, J. (1985). The CUSUM test of homogeneity with an
        application to spontaneous abortion epidemiology. Statistics in
        Medicine, 4(4), 469-488. (Coined the term "epidemic changepoint" for
        a transient regime change followed by a return to baseline.)

    Notes
    -----
    Using costs to generate transient scores via :class:`CostTransientScore`
    is significantly slower than using transient scores implemented directly,
    since the surrounding-baseline cost requires re-precomputing for each
    candidate inner interval.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.detectors import CircularBinarySegmentation
    >>> rng = np.random.default_rng(2)
    >>> X = np.concatenate([
    ...     rng.normal(0, 1, (40, 1)),
    ...     rng.normal(10, 1, (10, 1)),
    ...     rng.normal(0, 1, (40, 1)),
    ... ])
    >>> detector = CircularBinarySegmentation()
    >>> detector.fit(X).predict_segment_anomalies(X)
    array([[40, 50]])
    """

    _parameter_constraints = {
        "transient_score": [HasMethods(["fit", "precompute", "evaluate"]), None],
        "penalty": ["array-like", Interval(Real, 0, None, closed="left"), None],
        "penalty_scale": [Interval(Real, 0, None, closed="neither")],
        "agg": [StrOptions(set(USER_AGG_CHOICES))],
        "min_subinterval_length": [Interval(Integral, 1, None, closed="left")],
        "max_interval_length": [Interval(Integral, 2, None, closed="left"), None],
        "growth_factor": [Interval(Real, 1.0, 2.0, closed="right")],
    }

    def __init__(
        self,
        transient_score: BaseIntervalScorer | None = None,
        penalty: ArrayLike | float | None = None,
        penalty_scale: float = 2.0,
        agg: str = "sum",
        min_subinterval_length: int = 5,
        max_interval_length: int | None = None,
        growth_factor: float = 1.8,
    ):
        self.transient_score = transient_score
        self.penalty = penalty
        self.penalty_scale = penalty_scale
        self.agg = agg
        self.min_subinterval_length = min_subinterval_length
        self.max_interval_length = max_interval_length
        self.growth_factor = growth_factor

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get tags, propagating input constraints from the wrapped scorer."""
        tags = super().__sklearn_tags__()
        scorer_tags = _resolve_transient_score(self.transient_score).__sklearn_tags__()
        tags.input_tags = scorer_tags.input_tags
        tags.change_detector_tags.linear_trend_segment = (
            scorer_tags.interval_scorer_tags.linear_trend_segment
        )
        tags.change_detector_tags.calibration_strategy = "max_score"
        return tags

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the transient score to training data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training time series data.
        y : ArrayLike | None, default=None
            Ignored.

        Returns
        -------
        self : CircularBinarySegmentation
            Fitted detector.
        """
        X = validate_data(self, X, reset=True, ensure_2d=True)

        transient_score = _resolve_transient_score(self.transient_score)
        check_interval_scorer(
            transient_score,
            ensure_score_type=["transient_score"],
            caller_name=self.__class__.__name__,
            arg_name="transient_score",
        )
        self.transient_score_ = clone(transient_score).fit(X, y)

        self.penalty_ = resolve_penalty(
            self.transient_score_,
            self.penalty,
            self.penalty_scale,
            caller_name=self.__class__.__name__,
            scorer_param_name="transient_score",
        )
        self._agg_mode = resolve_aggregation(
            self.transient_score_,
            self.agg,
            self.penalty_,
            self.n_features_in_,
            caller_name=self.__class__.__name__,
            scorer_param_name="transient_score",
        )

        self.min_subinterval_length_ = max(
            self.min_subinterval_length, self.transient_score_.min_size
        )

        if self.n_samples_in_ < 2 * self.min_subinterval_length_:
            raise ValueError(
                f"`CircularBinarySegmentation` requires at least "
                f"2 * min_subinterval_length "
                f"(={2 * self.min_subinterval_length_}) samples to fit, got "
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
        """Run circular binary segmentation and return all outputs in a single pass.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyse.

        Returns
        -------
        result : dict with keys:

            ``"segment_anomalies"`` : np.ndarray of shape (n_anomalies, 2)
                Each row is ``[start, end)`` of a detected segment anomaly,
                sorted by start.
            ``"changepoints"`` : np.ndarray of shape (n_changepoints,)
                Sorted unique inner boundary indices of detected anomalies.
            ``"interval_starts"`` : np.ndarray
                Start indices of the seeded outer intervals evaluated.
            ``"interval_ends"`` : np.ndarray
                End indices of the seeded outer intervals evaluated.
            ``"interval_max_scores"`` : np.ndarray
                Maximum (aggregated, penalised) score within each outer interval.
            ``"interval_argmax_inner_starts"`` : np.ndarray
                Best inner-interval start per outer interval.
            ``"interval_argmax_inner_ends"`` : np.ndarray
                Best inner-interval end per outer interval.
        """
        check_is_fitted(self)

        max_scores, index = self.predict_scores(X, return_index=True)
        starts = index["starts"]
        ends = index["ends"]
        argmax_inner_starts = index["argmax_inner_starts"]
        argmax_inner_ends = index["argmax_inner_ends"]

        segments = greedy_segment_selection(
            max_scores, argmax_inner_starts, argmax_inner_ends, starts, ends
        )

        if len(segments) == 0:
            changepoints = np.empty(0, dtype=np.intp)
        else:
            n_samples = self.n_samples_in_
            boundaries = np.unique(segments)
            changepoints = boundaries[
                (boundaries > 0) & (boundaries < n_samples)
            ].astype(np.intp)

        return {
            "segment_anomalies": segments,
            "changepoints": changepoints,
            "interval_starts": starts,
            "interval_ends": ends,
            "interval_max_scores": max_scores,
            "interval_argmax_inner_starts": argmax_inner_starts,
            "interval_argmax_inner_ends": argmax_inner_ends,
        }

    def predict_segment_anomalies(self, X: ArrayLike) -> np.ndarray:
        """Detect anomalies as ``[start, end)`` intervals.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyse for anomalies.

        Returns
        -------
        anomalies : np.ndarray of shape (n_anomalies, 2)
            Each row is ``[start, end)`` of a detected anomaly, sorted by start.
        """
        return self.predict_all(X)["segment_anomalies"]

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return sorted anomaly boundary indices.

        Each anomaly interval ``[start, end)`` contributes two changepoints
        (``start`` and ``end``) at the regime transitions in and out of the
        anomalous segment.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyse.

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
        """Return the per-outer-interval scoring objective evaluated on ``X``.

        For each seeded outer interval, returns the maximum aggregated,
        penalised transient score over candidate inner ``[start, end)``
        sub-intervals. This is what :meth:`predict_segment_anomalies` reduces
        over via the greedy selection step, without the selection itself.

        For penalty calibration, use the free function
        :func:`skchange.new_api.tuning.unpenalised_scores`, which fits a
        clone of this detector with the penalty parameter set to zero and
        returns the resulting unpenalised scores.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to evaluate.
        return_index : bool, default=False
            If ``True``, also return a dict locating each score on the time
            axis. See the Returns section for the keys.

        Returns
        -------
        scores : np.ndarray of shape (n_outer,)
            Aggregated, penalised transient-score values, one per seeded outer
            interval, at the best inner sub-interval within that outer
            interval. Returned alone when ``return_index=False``.
        index : dict, optional
            Only returned when ``return_index=True``. Contains:

            - ``"starts"`` : np.ndarray of shape (n_outer,)
              Start indices of the seeded outer intervals.
            - ``"ends"`` : np.ndarray of shape (n_outer,)
              End indices of the seeded outer intervals.
            - ``"argmax_inner_starts"`` : np.ndarray of shape (n_outer,)
              Inner-interval start of the maximising candidate per outer interval.
            - ``"argmax_inner_ends"`` : np.ndarray of shape (n_outer,)
              Inner-interval end of the maximising candidate per outer interval.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True)

        (
            max_scores,
            argmax_inner_starts,
            argmax_inner_ends,
            starts,
            ends,
        ) = _score_circular_intervals(
            transient_score=self.transient_score_,
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
                "argmax_inner_starts": argmax_inner_starts,
                "argmax_inner_ends": argmax_inner_ends,
            }
        return max_scores
