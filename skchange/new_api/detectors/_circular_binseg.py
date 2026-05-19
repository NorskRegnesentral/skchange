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
from skchange.new_api.interval_scorers._penalised_score import PenalisedScore
from skchange.new_api.interval_scorers._transient_scores.l2_transient_score import (
    L2TransientScore,
)
from skchange.new_api.typing import ArrayLike, Self
from skchange.new_api.utils import SkchangeTags
from skchange.new_api.utils._numba import njit
from skchange.new_api.utils._param_validation import (
    HasMethods,
    Interval,
    _fit_context,
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
def greedy_anomaly_selection(
    penalised_scores: np.ndarray,
    anomaly_starts: np.ndarray,
    anomaly_ends: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> list[tuple[int, int]]:
    """Greedily select non-overlapping segment anomalies with positive score."""
    penalised_scores = penalised_scores.copy()
    anomalies = []
    while np.any(penalised_scores > 0):
        argmax = penalised_scores.argmax()
        anomaly_start = anomaly_starts[argmax]
        anomaly_end = anomaly_ends[argmax]
        anomalies.append((anomaly_start, anomaly_end))
        # Remove outer intervals that overlap with the detected segment anomaly.
        penalised_scores[(anomaly_end > starts) & (anomaly_start < ends)] = 0.0
    anomalies.sort()
    return anomalies


def _run_circular_binseg(
    transient_score: BaseIntervalScorer,
    X: np.ndarray,
    min_subinterval_length: int,
    max_interval_length: int,
    growth_factor: float,
) -> tuple[
    list[tuple[int, int]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Run the circular binary segmentation algorithm.

    Parameters
    ----------
    transient_score : BaseIntervalScorer
        A fitted, penalised transient score.
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
    inner_intervals : list of (int, int)
        Detected ``(start, end)`` inner intervals (transient changes).
    max_scores : np.ndarray
        Best (aggregated) penalised score for each outer interval.
    argmax_inner_starts : np.ndarray
        Inner-interval start of the best inner candidate per outer interval.
    argmax_inner_ends : np.ndarray
        Inner-interval end of the best inner candidate per outer interval.
    starts : np.ndarray
        Outer-interval start indices.
    ends : np.ndarray
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

    # Build the (outer_start, inner_start, inner_end, outer_end) specs for all
    # inner candidates across every outer interval and evaluate the transient
    # score in a single call. Same approach as in ``_run_seeded_binseg``.
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
    all_scores = transient_score.evaluate(cache, interval_specs)
    # Aggregate across feature columns when the score is multivariate.
    if all_scores.ndim == 2:
        all_scores = np.sum(all_scores, axis=1)
    all_scores = all_scores.reshape(-1)

    offsets = np.concatenate(([0], np.cumsum(n_inner)))
    for i in range(starts.size):
        interval_scores = all_scores[offsets[i] : offsets[i + 1]]
        argmax = int(np.argmax(interval_scores))
        max_scores[i] = interval_scores[argmax]
        inner_starts_i, inner_ends_i = inner_per_interval[i]
        argmax_inner_starts[i] = inner_starts_i[argmax]
        argmax_inner_ends[i] = inner_ends_i[argmax]

    inner_intervals = greedy_anomaly_selection(
        max_scores, argmax_inner_starts, argmax_inner_ends, starts, ends
    )
    return (
        inner_intervals,
        max_scores,
        argmax_inner_starts,
        argmax_inner_ends,
        starts,
        ends,
    )


def _resolve_transient_score(
    transient_score: BaseIntervalScorer | None,
    penalty_scale: float = 1.0,
) -> BaseIntervalScorer:
    """Return a penalised transient score, auto-wrapping if needed.

    Needed since default resolution must be done in both fit and
    ``__sklearn_tags__`` to ensure correct input tags are propagated.
    """
    transient_score = L2TransientScore() if transient_score is None else transient_score
    tags = transient_score.__sklearn_tags__().interval_scorer_tags
    if tags.penalised:
        return transient_score
    return PenalisedScore(transient_score, penalty_scale=penalty_scale)


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

    Parameters
    ----------
    transient_score : BaseIntervalScorer or None, default=None
        Transient score to use in the algorithm. Must be an instance of
        ``BaseIntervalScorer`` with ``score_type="transient_score"``. If the
        scorer is unpenalised it is automatically wrapped in
        :class:`PenalisedScore`. If ``None``, defaults to
        ``PenalisedScore(L2TransientScore())``.

        Wrap with :class:`PenalisedScore` explicitly to set a custom
        ``penalty``, e.g.:

        * ``L2TransientScore()`` -- auto-wrapped with default BIC penalty
        * ``PenalisedScore(CostTransientScore(GaussianCost()), penalty=10.0)``
          -- Gaussian cost-based transient score with fixed penalty
    penalty_scale : float, default=2.0
        Multiplicative factor on the default penalty of the auto-constructed
        :class:`PenalisedScore` wrapper. Applies only when ``transient_score``
        is ``None`` or an unpenalised scorer. Silently ignored when
        ``transient_score`` is already a penalised scorer; in that case the
        user-provided scorer owns its penalty. The default is larger than 1
        because CBS evaluates the score over a very large number of candidate
        ``(outer, inner)`` interval pairs, so a stricter penalty is needed to
        keep the family-wise false-positive rate low.
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
        Fitted penalised transient score.
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
        "transient_score": [HasMethods(["fit", "evaluate"]), None],
        "penalty_scale": [Interval(Real, 0, None, closed="neither")],
        "min_subinterval_length": [Interval(Integral, 1, None, closed="left")],
        "max_interval_length": [Interval(Integral, 2, None, closed="left"), None],
        "growth_factor": [Interval(Real, 1.0, 2.0, closed="right")],
    }

    def __init__(
        self,
        transient_score: BaseIntervalScorer | None = None,
        penalty_scale: float = 2.0,
        min_subinterval_length: int = 5,
        max_interval_length: int | None = None,
        growth_factor: float = 1.8,
    ):
        self.transient_score = transient_score
        self.penalty_scale = penalty_scale
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

        scorer = _resolve_transient_score(self.transient_score, self.penalty_scale)
        check_interval_scorer(
            scorer,
            ensure_score_type=["transient_score"],
            caller_name=self.__class__.__name__,
            arg_name="transient_score",
        )
        self.transient_score_ = clone(scorer).fit(X, y)

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
                Maximum (aggregated) penalised score within each outer interval.
            ``"interval_argmax_inner_starts"`` : np.ndarray
                Best inner-interval start per outer interval.
            ``"interval_argmax_inner_ends"`` : np.ndarray
                Best inner-interval end per outer interval.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True)

        (
            inner_intervals,
            max_scores,
            argmax_inner_starts,
            argmax_inner_ends,
            starts,
            ends,
        ) = _run_circular_binseg(
            self.transient_score_,
            X,
            self.min_subinterval_length_,
            self.max_interval_length_,
            self.growth_factor,
        )

        if len(inner_intervals) == 0:
            segment_anomalies = np.empty((0, 2), dtype=np.intp)
            changepoints = np.empty(0, dtype=np.intp)
        else:
            segment_anomalies = np.asarray(inner_intervals, dtype=np.intp)
            n_samples = X.shape[0]
            boundaries = np.unique(segment_anomalies)
            changepoints = boundaries[
                (boundaries > 0) & (boundaries < n_samples)
            ].astype(np.intp)

        return {
            "segment_anomalies": segment_anomalies,
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

    def predict_changepoints(self, X: ArrayLike) -> np.ndarray:
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
            Sorted unique inner boundary indices of detected anomalies.
            Empty array if no anomalies are detected.
        """
        return self.predict_all(X)["changepoints"]
