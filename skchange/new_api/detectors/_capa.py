"""CAPA: Collective and Point Anomaly detection algorithm."""

__author__ = ["Tveten"]

from numbers import Integral, Real

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.detectors._base import BaseChangeDetector
from skchange.new_api.interval_scorers._base import (
    BaseIntervalScorer,
    is_aggregated_score,
    is_penalised_score,
)
from skchange.new_api.interval_scorers._savings.l1_saving import L1Saving
from skchange.new_api.interval_scorers._savings.l2_saving import L2Saving
from skchange.new_api.penalties import linear_chi2_penalty
from skchange.new_api.types import ArrayLike, Self
from skchange.new_api.utils import SkchangeTags
from skchange.new_api.utils._param_validation import (
    HasMethods,
    Interval,
    _fit_context,
)
from skchange.new_api.utils._score_aggregation import (
    aggregate_and_penalise,
    resolve_aggregation,
    resolve_penalty,
)
from skchange.new_api.utils.validation import (
    check_interval_scorer,
    validate_data,
)


def _run_capa(
    segment_saving: BaseIntervalScorer,
    segment_cache: dict,
    segment_agg_mode: str,
    segment_penalty: float | np.ndarray | None,
    point_saving: BaseIntervalScorer,
    point_cache: dict,
    point_agg_mode: str,
    point_penalty: float | np.ndarray | None,
    n_samples: int,
    min_segment_length: int,
    max_segment_length: int,
    log_savings: bool = False,
) -> tuple:
    """Run the CAPA dynamic programming algorithm.

    For each timestep the DP compares (i) no anomaly, (ii) the best segment
    anomaly ``[start, t+1)`` for ``start ∈ admissible_starts`` and (iii) a
    point anomaly ``[t, t+1)``. Aggregation modes and penalties for both
    scorers come from :func:`resolve_aggregation` / :func:`resolve_penalty`;
    the penalty is ``None`` when the scorer is inherently penalised
    (``agg_mode == "passthrough"``). Admissible starts are pruned each step
    using the maximum penalty across features.

    When ``log_savings=True`` the per-evaluation savings and their
    ``(start, end)`` / ``t`` indices are also returned; otherwise the
    corresponding outputs are empty arrays.

    Returns
    -------
    opt_savings : np.ndarray of shape (n_samples,)
        Cumulative optimal savings at each timestep.
    opt_anomaly_starts : np.ndarray of shape (n_samples,)
        For each ``t``, start index of the optimal anomaly ending at ``t+1``,
        or ``NaN`` if no anomaly ends there.
    segment_savings_all, segment_starts_all, segment_ends_all : np.ndarray
        Penalised saving and ``[start, end)`` index for every evaluated
        segment interval (only populated when ``log_savings=True``).
    point_savings_all, point_indices_all : np.ndarray
        Penalised saving and sample index ``t`` for every evaluated point
        interval ``[t, t+1)`` (only populated when ``log_savings=True``).
    """
    opt_savings = np.zeros(n_samples + 1)
    opt_anomaly_starts = np.full(n_samples, np.nan)
    starts = np.empty(0, dtype=np.intp)

    segment_savings_chunks: list[np.ndarray] = []
    segment_starts_chunks: list[np.ndarray] = []
    segment_ends_chunks: list[np.ndarray] = []
    point_savings_list: list[float] = []
    point_indices_list: list[int] = []

    # Pruning requires an upper bound on the segment penalty value.
    if segment_penalty is not None:
        max_segment_penalty = float(np.max(np.atleast_1d(segment_penalty)))
    elif hasattr(segment_saving, "penalty_"):
        max_segment_penalty = float(np.max(np.atleast_1d(segment_saving.penalty_)))
    else:
        max_segment_penalty = np.inf  # Don't prune when the penalty is unknown.

    for t in range(min_segment_length - 1, n_samples):
        # Extend the admissible segment starts by one at each step.
        starts = np.append(starts, np.intp(t - min_segment_length + 1))

        # Evaluate all candidate segment anomaly intervals [start, t+1).
        ends = np.full(len(starts), t + 1, dtype=np.intp)
        intervals = np.column_stack((starts, ends))
        raw_segment_scores = segment_saving.evaluate(segment_cache, intervals)
        segment_savings = aggregate_and_penalise(
            raw_segment_scores, segment_agg_mode, segment_penalty
        )
        if log_savings:
            segment_savings_chunks.append(segment_savings)
            segment_starts_chunks.append(starts.copy())
            segment_ends_chunks.append(ends)
        candidate_savings = opt_savings[starts] + segment_savings
        best_segment_idx = int(np.argmax(candidate_savings))
        opt_segment_saving = candidate_savings[best_segment_idx]
        opt_segment_start = starts[best_segment_idx]

        # Evaluate point anomaly [t, t+1).
        raw_point_score = point_saving.evaluate(point_cache, np.array([[t, t + 1]]))
        point_saving_value = float(
            aggregate_and_penalise(raw_point_score, point_agg_mode, point_penalty)[0]
        )
        if log_savings:
            point_savings_list.append(point_saving_value)
            point_indices_list.append(t)
        opt_point_saving = float(opt_savings[t]) + point_saving_value

        # Choose the best option: no anomaly, segment anomaly, or point anomaly.
        options = np.array(
            [float(opt_savings[t]), opt_segment_saving, opt_point_saving]
        )
        best = int(np.argmax(options))
        opt_savings[t + 1] = options[best]
        if best == 1:
            opt_anomaly_starts[t] = opt_segment_start
        elif best == 2:
            opt_anomaly_starts[t] = t

        # Prune starts that can no longer improve on the current optimal saving or
        # that would form segments longer than max_segment_length.
        prune = (candidate_savings + max_segment_penalty <= opt_savings[t + 1]) | (
            starts < t - max_segment_length + 2
        )
        starts = starts[~prune]

    if log_savings:
        segment_savings_all = np.concatenate(segment_savings_chunks)
        segment_starts_all = np.concatenate(segment_starts_chunks).astype(np.intp)
        segment_ends_all = np.concatenate(segment_ends_chunks).astype(np.intp)
    else:
        segment_savings_all = np.empty(0, dtype=float)
        segment_starts_all = np.empty(0, dtype=np.intp)
        segment_ends_all = np.empty(0, dtype=np.intp)
    point_savings_all = np.asarray(point_savings_list, dtype=float)
    point_indices_all = np.asarray(point_indices_list, dtype=np.intp)

    return (
        opt_savings[1:],
        opt_anomaly_starts,
        segment_savings_all,
        segment_starts_all,
        segment_ends_all,
        point_savings_all,
        point_indices_all,
    )


def _extract_anomalies(
    opt_anomaly_starts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract segment and point anomaly intervals from the DP result.

    Scans ``opt_anomaly_starts`` backwards, jumping over identified segments
    to avoid double-counting. Returns ``(segments, points)`` sorted by start
    index; segments have shape ``(k, 2)`` with rows ``[start, end)`` and
    length >= 2.
    """
    n = opt_anomaly_starts.size
    segment_anomalies: list[list[int]] = []
    point_anomalies: list[int] = []

    i = n - 1
    while i >= 0:
        start_i = opt_anomaly_starts[i]
        size = i - start_i + 1  # NaN when no anomaly → neither condition fires
        if size > 1:
            segment_anomalies.append([int(start_i), i + 1])
            i = int(start_i)  # jump to segment start, then decrement below
        elif size == 1:
            point_anomalies.append(i)
        i -= 1

    # Lists are built in descending index order; reversing gives ascending.
    segment_anomalies_arr = (
        np.array(segment_anomalies[::-1], dtype=np.intp)
        if segment_anomalies
        else np.empty((0, 2), dtype=np.intp)
    )
    point_anomalies_arr = np.array(point_anomalies[::-1], dtype=np.intp)
    return segment_anomalies_arr, point_anomalies_arr


def _get_changed_features(
    saving: BaseIntervalScorer,
    cache: dict,
    penalty: float | np.ndarray | None,
    intervals: np.ndarray,
) -> list[np.ndarray] | None:
    """Identify which features are anomalous for each detected interval.

    For each interval, evaluates the per-feature unpenalised saving, sorts
    features by saving (descending), and returns the prefix that maximises
    ``cumsum(sorted_savings) - penalty``.

    Only applicable when ``saving`` is an unpenalised, non-aggregated scorer
    (returns per-feature scores) and ``penalty`` is an array-valued penalty
    with one entry per feature. Returns ``None`` otherwise (penalised or
    aggregated saving, or scalar ``penalty``).
    """
    is_constant_penalty = np.asarray(penalty).ndim == 0
    if is_penalised_score(saving) or is_aggregated_score(saving) or is_constant_penalty:
        return None
    if len(intervals) == 0:
        return []

    penalty_values = np.asarray(penalty).reshape(-1)
    all_savings = saving.evaluate(cache, np.asarray(intervals))
    penalty_resized = np.resize(penalty_values, all_savings.shape[1])

    changed = []
    for saving_values in all_savings:
        saving_order = np.argsort(-saving_values)  # descending
        penalised_savings = np.cumsum(saving_values[saving_order]) - penalty_resized
        best_k = int(np.argmax(penalised_savings)) + 1
        changed.append(saving_order[:best_k].astype(np.intp))
    return changed


def _resolve_segment_saving(
    saving: BaseIntervalScorer | None,
) -> BaseIntervalScorer:
    """Return the segment saving, defaulting to :class:`L2Saving` when ``None``.

    Used in both fit() and __sklearn_tags__() so that input tags are propagated
    consistently whether or not a saving is explicitly provided.
    """
    return L2Saving() if saving is None else saving


def _resolve_point_saving(
    point_saving: BaseIntervalScorer | None,
    segment_saving_: BaseIntervalScorer,
) -> BaseIntervalScorer:
    """Return the point saving, defaulting based on the segment saving.

    When ``point_saving`` is ``None``:

    * If ``segment_saving_`` has ``min_size == 1`` and is not itself a
      penalised scorer, the point saving is a fresh clone of
      ``segment_saving_``.
    * Otherwise the default is :class:`L1Saving`.

    Penalty and aggregation are applied externally by :func:`resolve_penalty`
    and :func:`resolve_aggregation` in ``fit``.
    """
    if point_saving is not None:
        return point_saving
    if segment_saving_.min_size == 1 and not is_penalised_score(segment_saving_):
        return clone(segment_saving_)
    return L1Saving()


class CAPA(BaseChangeDetector):
    """Collective and Point Anomaly (CAPA) detection algorithm.

    An efficient implementation of the CAPA family of algorithms for anomaly detection
    [1]_ [2]_. Detects contiguous anomalous segments (collective anomalies) and
    isolated anomalous samples (point anomalies) via a dynamic programming
    formulation based on a penalised saving.

    Standard usage is to pass unpenalised savings (or ``None``) and configure
    the penalty via ``segment_penalty`` / ``point_penalty`` / ``penalty_scale``
    / ``agg``. Already-penalised scorers are also accepted; in that case the
    scorer owns its own penalty/aggregation and the corresponding detector
    parameters are ignored.

    Parameters
    ----------
    segment_saving : BaseIntervalScorer or None, default=None
        Saving for segment anomaly detection. Must be an instance of
        ``BaseIntervalScorer`` with ``score_type="saving"``. If ``None``,
        defaults to :class:`L2Saving`.
    point_saving : BaseIntervalScorer or None, default=None
        Saving for point anomaly detection. Must be an instance of
        ``BaseIntervalScorer`` with ``score_type="saving"`` and
        ``min_size == 1``. If ``None``, defaults to a clone of
        ``segment_saving`` when ``segment_saving.min_size == 1`` (and
        ``segment_saving`` is not itself penalised), otherwise
        :class:`L1Saving`.
    segment_penalty : float, array-like of shape (n_features,) or None, default=None
        Penalty subtracted from the aggregated segment saving; a candidate is
        accepted only when the result is positive.

        - ``float``: scalar penalty (summed saving across features).
        - ``array-like`` of length ``n_features``, non-decreasing: element
          ``i`` is the penalty for ``i+1`` jointly affected features; CAPA
          picks the ``k`` largest feature savings maximising
          ``sum(top_k) - penalty[k-1]`` (handles sparse anomalies).
        - ``None``: uses ``segment_saving.get_default_penalty()``.

        Ignored when ``segment_saving`` is already penalised.
    point_penalty : float, array-like of shape (n_features,) or None, default=None
        Same semantics as ``segment_penalty`` but for point anomalies.
        Defaults to ``2 * linear_chi2_penalty(n_samples, n_features)`` —
        twice the segment default — to prioritise segment anomalies over
        isolated points. Ignored when ``point_saving`` is already penalised.
    penalty_scale : float, default=1.0
        Positive multiplier applied to both ``segment_penalty`` and
        ``point_penalty``. A single tuning knob that preserves the shape
        of array penalties.
    min_segment_length : int or None, default=None
        Minimum segment anomaly length. Defaults to
        ``2 * segment_saving.min_size`` — a finite-sample safety floor
        against spurious short segments from scale-estimating savings (e.g.
        Gaussian, Laplace). Must be at least ``segment_saving.min_size``.
    max_segment_length : int or None, default=None
        Maximum number of samples in a segment anomaly. Defaults to
        ``n_samples // 2`` when ``None``, with a minimum of ``min_segment_length``.
    include_point_anomalies : bool, default=False
        If ``True``, detected point anomalies are included alongside segment
        anomalies in the output of ``predict``, ``predict_segment_anomalies``,
        ``predict_changepoints``, and ``predict_scores`` treated as single-sample
        intervals. Point anomalies are always available via ``predict_all`` regardless
        of this setting.

    Attributes
    ----------
    segment_saving_ : BaseIntervalScorer
        Fitted segment saving scorer (the unpenalised scorer, or the
        user-supplied penalised scorer).
    point_saving_ : BaseIntervalScorer
        Fitted point saving scorer.
    segment_penalty_ : float, np.ndarray or None
        Effective segment penalty used at detection time (resolved base
        penalty multiplied by ``penalty_scale``). ``None`` when
        ``segment_saving`` is inherently penalised.
    point_penalty_ : float, np.ndarray or None
        Effective point penalty. ``None`` when ``point_saving`` is inherently
        penalised.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). A linear time method
       for the detection of collective and point anomalies. Statistical Analysis and
       DataMining: The ASA Data Science Journal, 15(4), 494-508.

    .. [2] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       collective and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.detectors import CAPA
    >>> rng = np.random.default_rng(2)
    >>> X = np.concatenate([rng.normal(0, 1, (100, 1)),
    ...                     rng.normal(10, 1, (20, 1)),
    ...                     rng.normal(0, 1, (100, 1))])
    >>> detector = CAPA()
    >>> detector.fit(X).predict_segment_anomalies(X)
    array([[100, 120]])
    """

    # CAPA carries separate segment and point penalties rather than a single
    # scalar base, so the closed-form ``max_score`` path does not apply;
    # calibration uses the universal ``detection_count`` bisection.
    _calibration_strategy = "detection_count"

    _parameter_constraints = {
        "segment_saving": [HasMethods(["fit", "precompute", "evaluate"]), None],
        "point_saving": [HasMethods(["fit", "precompute", "evaluate"]), None],
        "segment_penalty": [
            "array-like",
            Interval(Real, 0, None, closed="left"),
            None,
        ],
        "point_penalty": [
            "array-like",
            Interval(Real, 0, None, closed="left"),
            None,
        ],
        "penalty_scale": [Interval(Real, 0, None, closed="neither")],
        "min_segment_length": [Interval(Integral, 2, None, closed="left"), None],
        "max_segment_length": [Interval(Integral, 2, None, closed="left"), None],
        "include_point_anomalies": ["boolean"],
    }

    def __init__(
        self,
        segment_saving: BaseIntervalScorer | None = None,
        point_saving: BaseIntervalScorer | None = None,
        segment_penalty: ArrayLike | float | None = None,
        point_penalty: ArrayLike | float | None = None,
        penalty_scale: float = 1.0,
        min_segment_length: int | None = None,
        max_segment_length: int | None = None,
        include_point_anomalies: bool = False,
    ):
        self.segment_saving = segment_saving
        self.point_saving = point_saving
        self.segment_penalty = segment_penalty
        self.point_penalty = point_penalty
        self.penalty_scale = penalty_scale
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.include_point_anomalies = include_point_anomalies

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get tags, propagating input constraints from the segment saving."""
        tags = super().__sklearn_tags__()
        scorer_tags = _resolve_segment_saving(self.segment_saving).__sklearn_tags__()
        tags.input_tags = scorer_tags.input_tags
        tags.change_detector_tags.linear_trend_segment = (
            scorer_tags.interval_scorer_tags.linear_trend_segment
        )
        return tags

    def _resolve_aggregation(
        self,
        scorer: BaseIntervalScorer,
        penalty: ArrayLike | float | None,
        scorer_param_name: str,
    ) -> tuple[float | np.ndarray | None, str]:
        """Resolve ``(penalty, agg_mode)`` for one fitted scorer."""
        resolved_penalty = resolve_penalty(
            scorer,
            penalty,
            self.penalty_scale,
            caller_name=type(self).__name__,
            scorer_param_name=scorer_param_name,
        )
        agg_mode = resolve_aggregation(
            scorer,
            "sum",
            resolved_penalty,
            self.n_features_in_,
            caller_name=type(self).__name__,
            scorer_param_name=scorer_param_name,
        )
        return resolved_penalty, agg_mode

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit both savings to training data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training time series data.
        y : None
            Ignored.

        Returns
        -------
        self : CAPA
            Fitted detector.
        """
        X = validate_data(self, X, reset=True, ensure_2d=True)

        segment_saving = _resolve_segment_saving(self.segment_saving)
        check_interval_scorer(
            segment_saving,
            ensure_score_type=["saving"],
            caller_name=self.__class__.__name__,
            arg_name="segment_saving",
        )
        self.segment_saving_ = clone(segment_saving).fit(X, y)

        point_saving = _resolve_point_saving(self.point_saving, self.segment_saving_)
        check_interval_scorer(
            point_saving,
            ensure_score_type=["saving"],
            caller_name=self.__class__.__name__,
            arg_name="point_saving",
        )
        self.point_saving_ = clone(point_saving).fit(X, y)
        if self.point_saving_.min_size > 1:
            raise ValueError(
                f"`point_saving` must have min_size == 1, "
                f"got min_size={self.point_saving_.min_size}."
            )

        self.segment_penalty_, self._segment_agg_mode = self._resolve_aggregation(
            self.segment_saving_, self.segment_penalty, "segment_saving"
        )
        # The point penalty defaults to ``2 * linear_chi2_penalty`` — twice the segment
        # default — to prioritise segment anomalies over point anomalies.
        if self.point_penalty is None and not is_penalised_score(self.point_saving_):
            point_penalty_base = 2 * linear_chi2_penalty(
                self.n_samples_in_, self.n_features_in_
            )
        else:
            point_penalty_base = self.point_penalty
        self.point_penalty_, self._point_agg_mode = self._resolve_aggregation(
            self.point_saving_, point_penalty_base, "point_saving"
        )

        min_size = self.segment_saving_.min_size
        if self.min_segment_length is None:
            self._min_segment_length = 2 * min_size
        elif self.min_segment_length < min_size:
            raise ValueError(
                f"`min_segment_length={self.min_segment_length}` is less than "
                f"`segment_saving.min_size={min_size}`. "
                f"Set `min_segment_length=None` to use `min_size` as the default."
            )
        else:
            self._min_segment_length = self.min_segment_length

        self._max_segment_length = (
            max(self.n_samples_in_ // 2, self._min_segment_length)
            if self.max_segment_length is None
            else self.max_segment_length
        )
        if self._min_segment_length > self._max_segment_length:
            raise ValueError(
                f"min_segment_length ({self._min_segment_length}) must not "
                f"exceed max_segment_length ({self._max_segment_length})."
            )

        return self

    def _run(self, X: np.ndarray, log_savings: bool = False) -> tuple:
        """Run CAPA on ``X`` with the fitted state.

        Validates ``X``, precomputes scorer caches, and invokes
        :func:`_run_capa`. Returns the raw DP outputs followed by the
        scorer caches so callers that need per-anomaly feature extraction
        can reuse them without recomputing.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Validated time series.
        log_savings : bool, optional
            Forwarded to :func:`_run_capa` to control whether the
            per-evaluation savings logs are populated.
        """
        if self._min_segment_length > X.shape[0]:
            raise ValueError(
                f"`min_segment_length` ({self._min_segment_length}) cannot be "
                f"larger than the number of samples ({X.shape[0]})."
            )

        segment_cache = self.segment_saving_.precompute(X)
        point_cache = self.point_saving_.precompute(X)

        return (
            *_run_capa(
                self.segment_saving_,
                segment_cache,
                self._segment_agg_mode,
                self.segment_penalty_,
                self.point_saving_,
                point_cache,
                self._point_agg_mode,
                self.point_penalty_,
                X.shape[0],
                self._min_segment_length,
                self._max_segment_length,
                log_savings=log_savings,
            ),
            segment_cache,
            point_cache,
        )

    def predict_all(self, X: ArrayLike) -> dict:
        """Detect anomalies, returning all outputs in a single pass.

        This is the primary computation method. All other ``predict_*`` methods
        derive their results from this one.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyse for anomalies.

        Returns
        -------
        result : dict with keys:

            ``"segment_anomalies"`` : np.ndarray of shape (n_segment_anomalies, 2)
                Each row is ``[start, end)`` of a contiguous segment anomaly.
            ``"point_anomalies"`` : np.ndarray of shape (n_point_anomalies,)
                Sorted sample indices of point anomalies.
            ``"cumulative_optimal_savings"`` : np.ndarray of shape (n_samples,)
                Cumulative optimal savings from the dynamic programme.
            ``"segment_anomaly_features"`` : list of np.ndarray or None
                One array per detected segment anomaly with the 0-based
                feature indices identified as changed, ordered from strongest
                to weakest evidence. ``None`` when ``segment_saving`` is
                penalised or aggregated, or when ``segment_penalty_`` is
                scalar (i.e. no per-feature attribution is possible).
            ``"point_anomaly_features"`` : list of np.ndarray or None
                Same as above, but for point anomalies (driven by
                ``point_saving`` and ``point_penalty_``).
            ``"segment_savings"``, ``"segment_starts"``, ``"segment_ends"`` : np.ndarray
                Penalised saving and ``[start, end)`` index for every
                segment interval the DP evaluated. With pruning, the
                evaluated set depends on ``segment_penalty``.
            ``"point_savings"``, ``"point_indices"`` : np.ndarray
                Penalised saving and sample index ``t`` for every evaluated
                point interval ``[t, t+1)``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True)

        (
            opt_savings,
            opt_anomaly_starts,
            segment_savings,
            segment_starts,
            segment_ends,
            point_savings,
            point_indices,
            segment_cache,
            point_cache,
        ) = self._run(X, log_savings=True)
        segment_anomalies, point_anomalies = _extract_anomalies(opt_anomaly_starts)

        # Changed features computed separately for segments and points.
        # Points are expanded to [t, t+1) intervals for evaluation.
        segment_anomaly_features = _get_changed_features(
            self.segment_saving_,
            segment_cache,
            self.segment_penalty_,
            segment_anomalies,
        )
        point_intervals = (
            np.column_stack([point_anomalies, point_anomalies + 1])
            if len(point_anomalies)
            else np.empty((0, 2), dtype=np.intp)
        )
        point_anomaly_features = _get_changed_features(
            self.point_saving_,
            point_cache,
            self.point_penalty_,
            point_intervals,
        )

        return {
            "segment_anomalies": segment_anomalies,
            "point_anomalies": point_anomalies,
            "cumulative_optimal_savings": opt_savings,
            "segment_anomaly_features": segment_anomaly_features,
            "point_anomaly_features": point_anomaly_features,
            "segment_savings": segment_savings,
            "segment_starts": segment_starts,
            "segment_ends": segment_ends,
            "point_savings": point_savings,
            "point_indices": point_indices,
        }

    def predict_segment_anomalies(self, X: ArrayLike) -> np.ndarray:
        """Detect anomalies as ``[start, end)`` intervals.

        When ``include_point_anomalies=True``, point anomalies are appended as
        single-sample intervals and the result is sorted by start index.
        Use ``predict_all`` to access segment and point anomalies separately.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyse for anomalies.

        Returns
        -------
        anomalies : np.ndarray of shape (n_anomalies, 2)
            Each row is ``[start, end)`` of a detected anomaly, sorted by start.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True)
        _, opt_anomaly_starts, *_ = self._run(X, log_savings=False)
        segment_anomalies, point_anomalies = _extract_anomalies(opt_anomaly_starts)
        if not self.include_point_anomalies or len(point_anomalies) == 0:
            return segment_anomalies
        point_intervals = np.column_stack([point_anomalies, point_anomalies + 1])
        all_intervals = (
            np.vstack([segment_anomalies, point_intervals])
            if len(segment_anomalies)
            else point_intervals
        )
        return all_intervals[np.argsort(all_intervals[:, 0])]

    def predict_changepoints(self, X: ArrayLike) -> np.ndarray:
        """Return sorted anomaly boundary indices.

        Each anomaly interval ``[start, end)`` contributes two changepoints:
        ``start`` (regime transitions to anomalous) and ``end`` (back to normal).

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyse.

        Returns
        -------
        changepoints : np.ndarray of shape (n_changepoints,)
            Sorted unique inner boundary indices of detected anomalies.
            When ``include_point_anomalies=True``, point anomaly indices are
            also included. Use ``predict_all`` to access them separately.
        """
        anomalies = self.predict_segment_anomalies(X)
        if len(anomalies) == 0:
            return np.empty(0, dtype=np.intp)
        n_samples = validate_data(self, X, reset=False, ensure_2d=True).shape[0]
        boundaries = np.unique(anomalies)
        return boundaries[(boundaries > 0) & (boundaries < n_samples)].astype(np.intp)

    def predict_scores(
        self,
        X: ArrayLike,
        return_index: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return the penalised savings at every interval the CAPA DP evaluated.

        Concatenates the penalised savings for every ``[start, end)`` segment
        interval and, when ``include_point_anomalies=True``, every
        single-sample point interval that the dynamic programme actually
        visited. With pruning enabled, the set of evaluated segment intervals
        depends on the current ``segment_penalty_``; use
        :func:`skchange.new_api.tuning.unpenalised_scores` (with the penalty
        zeroed) for an unpruned grid.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to evaluate.
        return_index : bool, default=False
            If ``True``, also return a dict locating each score on the time
            axis. See the Returns section for the keys.

        Returns
        -------
        scores : np.ndarray of shape (n_evals,)
            Penalised savings, segment intervals first then (if
            ``include_point_anomalies=True``) point intervals.
            Returned alone when ``return_index=False``.
        index : dict, optional
            Only returned when ``return_index=True``. Contains:

            - ``"starts"`` : np.ndarray of shape (n_evals,)
              Start index of each evaluated interval. For point savings the
              start equals the sample index ``t``.
            - ``"ends"`` : np.ndarray of shape (n_evals,)
              End index of each evaluated interval. For point savings the end
              equals ``t + 1``.
        """
        result = self.predict_all(X)
        if self.include_point_anomalies:
            scores = np.concatenate(
                [result["segment_savings"], result["point_savings"]]
            )
        else:
            scores = result["segment_savings"]
        if return_index:
            if self.include_point_anomalies:
                starts = np.concatenate(
                    [result["segment_starts"], result["point_indices"]]
                )
                ends = np.concatenate(
                    [result["segment_ends"], result["point_indices"] + 1]
                )
            else:
                starts = result["segment_starts"]
                ends = result["segment_ends"]
            return scores, {"starts": starts, "ends": ends}
        return scores

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Detect anomalies, returning per-sample segment labels.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyse.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Integer labels: ``0`` for normal samples, ``1, ..., K`` for each
            detected anomaly in chronological order. When
            ``include_point_anomalies=True``, point anomalies are included as
            single-sample intervals and numbered together with segment anomalies.
        """
        intervals = self.predict_segment_anomalies(X)
        n_samples = validate_data(self, X, reset=False, ensure_2d=True).shape[0]
        labels = np.zeros(n_samples, dtype=np.intp)
        for label, (start, end) in enumerate(intervals, start=1):
            labels[start:end] = label
        return labels
