"""CAPA: Collective and Point Anomaly detection algorithm."""

from numbers import Integral

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.detectors._base import BaseChangeDetector
from skchange.new_api.interval_scorers._base import BaseIntervalScorer
from skchange.new_api.interval_scorers._penalised_score import PenalisedScore
from skchange.new_api.interval_scorers._savings._l2_saving import L2Saving
from skchange.new_api.typing import ArrayLike, Self
from skchange.new_api.utils import SkchangeTags
from skchange.new_api.utils._param_validation import HasMethods, Interval, _fit_context
from skchange.new_api.utils.validation import check_interval_scorer, validate_data


def _resolve_saving(saving: BaseIntervalScorer | None) -> BaseIntervalScorer:
    """Return saving or the default PenalisedScore(L2Saving()).

    Used in both fit() and __sklearn_tags__() so that input tags are propagated
    consistently whether or not a saving is explicitly provided.
    """
    return saving if saving is not None else PenalisedScore(L2Saving())


def _run_capa(
    segment_scorer: BaseIntervalScorer,
    segment_cache: dict,
    point_scorer: BaseIntervalScorer,
    point_cache: dict,
    n_samples: int,
    min_segment_length: int,
    max_segment_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the CAPA dynamic programming algorithm.

    Parameters
    ----------
    segment_scorer : BaseIntervalScorer
        Fitted penalised saving for segment anomalies.
    segment_cache : dict
        Precomputed cache from segment_scorer.precompute(X).
    point_scorer : BaseIntervalScorer
        Fitted penalised saving for point anomalies.
    point_cache : dict
        Precomputed cache from point_scorer.precompute(X).
    n_samples : int
        Number of samples.
    min_segment_length : int
        Minimum segment anomaly length.
    max_segment_length : int
        Maximum segment anomaly length.

    Returns
    -------
    opt_savings : np.ndarray of shape (n_samples,)
        Cumulative optimal savings at each timestep.
    opt_anomaly_starts : np.ndarray of shape (n_samples,)
        For each timestep t, the start index of the optimal anomaly ending at t+1,
        or NaN if no anomaly ends there.
    """
    opt_savings = np.zeros(n_samples + 1)
    opt_anomaly_starts = np.full(n_samples, np.nan)
    starts = np.empty(0, dtype=np.intp)

    # Pruning requires knowing the maximum possible penalty value for a segment.
    max_segment_penalty = (
        float(np.max(segment_scorer.penalty_))
        if hasattr(segment_scorer, "penalty_")
        else 0.0
    )

    for t in range(min_segment_length - 1, n_samples):
        # Extend the admissible segment starts by one at each step.
        starts = np.append(starts, np.intp(t - min_segment_length + 1))

        # Evaluate all candidate segment anomaly intervals [start, t+1).
        ends = np.full(len(starts), t + 1, dtype=np.intp)
        intervals = np.column_stack((starts, ends))
        segment_savings = segment_scorer.evaluate(segment_cache, intervals).reshape(-1)
        candidate_savings = opt_savings[starts] + segment_savings
        best_segment_idx = int(np.argmax(candidate_savings))
        opt_segment_saving = candidate_savings[best_segment_idx]
        opt_segment_start = starts[best_segment_idx]

        # Evaluate point anomaly [t, t+1).
        point_saving = float(
            point_scorer.evaluate(point_cache, np.array([[t, t + 1]])).reshape(-1)[0]
        )
        opt_point_saving = float(opt_savings[t]) + point_saving

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

    return opt_savings[1:], opt_anomaly_starts


def _extract_anomalies(
    opt_anomaly_starts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract segment and point anomaly intervals from the DP result.

    Scans backwards through the DP result, jumping over identified segment anomalies
    to avoid double-counting.

    Parameters
    ----------
    opt_anomaly_starts : np.ndarray of shape (n_samples,)
        DP result: for each t, start index of an anomaly ending at t+1, or NaN.

    Returns
    -------
    segment_anomalies : np.ndarray of shape (n_segment_anomalies, 2)
        Each row is [start, end) for a contiguous segment anomaly (length >= 2),
        sorted by start index.
    point_anomalies : np.ndarray of shape (n_point_anomalies,)
        Sorted sample indices of point anomalies.
    """
    n = opt_anomaly_starts.size
    segment_anomalies = []
    point_anomalies = []

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

    segment_anomalies_arr = (
        np.array(sorted(segment_anomalies), dtype=np.intp)
        if segment_anomalies
        else np.empty((0, 2), dtype=np.intp)
    )
    point_anomalies_arr = (
        np.array(sorted(point_anomalies), dtype=np.intp)
        if point_anomalies
        else np.empty(0, dtype=np.intp)
    )
    return segment_anomalies_arr, point_anomalies_arr


def _get_changed_features(
    penalised_saving: PenalisedScore,
    cache: dict,
    intervals: np.ndarray,
) -> list[np.ndarray]:
    """Identify which features are anomalous for each detected interval.

    For each anomaly interval the inner (unpenalised) saving is evaluated
    per feature.  Features are sorted by saving in descending order and
    the optimal subset is the prefix that maximises
    ``cumsum(sorted_savings) - penalty``.

    Only meaningful when ``penalised_saving`` is a :class:`PenalisedScore` whose
    inner scorer returns per-feature (non-aggregated) scores.  Returns an empty
    array of features for every interval when that condition is not met.

    Parameters
    ----------
    penalised_saving : PenalisedScore
        Fitted penalised saving.
    cache : dict
        Precomputed cache from ``penalised_saving.precompute(X)``.
    intervals : np.ndarray of shape (n_anomalies, 2)
        Anomaly intervals ``[start, end)``.

    Returns
    -------
    changed_features : list of np.ndarray
        One array per anomaly.  Each array contains the 0-based indices of the
        features identified as changed, sorted by evidence (strongest first).
    """
    if not isinstance(penalised_saving, PenalisedScore):
        return [np.empty(0, dtype=np.intp) for _ in range(len(intervals))]

    inner_scorer = penalised_saving.scorer_
    if inner_scorer.__sklearn_tags__().interval_scorer_tags.aggregated:
        return [np.empty(0, dtype=np.intp) for _ in range(len(intervals))]

    penalty_values = np.asarray(penalised_saving.penalty_).reshape(-1)
    changed = []
    for start, end in intervals:
        interval_spec = np.array([[start, end]])
        saving_values = inner_scorer.evaluate(cache, interval_spec)[0]
        saving_order = np.argsort(-saving_values)  # descending
        penalised_savings = np.cumsum(saving_values[saving_order]) - np.resize(
            penalty_values, len(saving_values)
        )
        best_k = int(np.argmax(penalised_savings)) + 1
        changed.append(saving_order[:best_k].astype(np.intp))
    return changed


class CAPA(BaseChangeDetector):
    """Collective and Point Anomaly (CAPA) detection algorithm.

    An efficient implementation of the CAPA family of algorithms for anomaly detection
    [1]_ [2]_. Detects contiguous anomalous segments (collective anomalies) and
    isolated anomalous samples (point anomalies) via a dynamic programming
    formulation based on a penalised saving.

    Users must supply a **penalised** saving.
    Use :class:`PenalisedScore` to compose any unpenalised saving or cost scorer with
    the desired penalty (e.g. ``PenalisedScore(L2Saving())``).

    Parameters
    ----------
    segment_saving : BaseIntervalScorer or None, default=None
        Penalised saving for segment anomaly detection.
        Must have ``interval_scorer_tags.penalised = True``.
        If ``None``, defaults to ``PenalisedScore(L2Saving())``.
    point_saving : BaseIntervalScorer or None, default=None
        Penalised saving for point anomaly detection.
        Must have ``interval_scorer_tags.penalised = True`` and ``min_size == 1``.
        If ``None``, defaults to ``PenalisedScore(L2Saving())``.
    min_segment_length : int, default=2
        Minimum number of samples in a segment anomaly. The effective minimum
        is ``max(min_segment_length, segment_saving.min_size)`` after fitting.
    max_segment_length : int, default=1000
        Maximum number of samples in a segment anomaly.
    include_point_anomalies : bool, default=False
        If ``True``, detected point anomalies are included alongside segment
        anomalies in the output of ``predict``, ``predict_segment_anomalies``,
        and ``predict_changepoints``, treated as single-sample intervals.
        Point anomalies are always available via ``predict_all`` regardless of
        this setting.

    Attributes
    ----------
    segment_saving_ : BaseIntervalScorer
        Fitted segment saving scorer.
    point_saving_ : BaseIntervalScorer
        Fitted point saving scorer.

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
    >>> from skchange.new_api.interval_scorers import PenalisedScore, L2Saving
    >>> rng = np.random.default_rng(2)
    >>> X = np.concatenate([rng.normal(0, 1, (100, 1)),
    ...                     rng.normal(10, 1, (20, 1)),
    ...                     rng.normal(0, 1, (100, 1))])
    >>> detector = CAPA()
    >>> detector.fit(X).predict_segment_anomalies(X)
    array([[100, 120]])
    """

    _parameter_constraints = {
        "segment_saving": [HasMethods(["fit", "precompute", "evaluate"]), None],
        "point_saving": [HasMethods(["fit", "precompute", "evaluate"]), None],
        "min_segment_length": [Interval(Integral, 2, None, closed="left")],
        "max_segment_length": [Interval(Integral, 2, None, closed="left")],
        "include_point_anomalies": ["boolean"],
    }

    def __init__(
        self,
        segment_saving: BaseIntervalScorer | None = None,
        point_saving: BaseIntervalScorer | None = None,
        min_segment_length: int = 2,
        max_segment_length: int = 1000,
        include_point_anomalies: bool = False,
    ):
        self.segment_saving = segment_saving
        self.point_saving = point_saving
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.include_point_anomalies = include_point_anomalies

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get tags, propagating input constraints from the segment saving."""
        tags = super().__sklearn_tags__()
        tags.input_tags = (
            _resolve_saving(self.segment_saving).__sklearn_tags__().input_tags
        )
        return tags

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

        segment_saving = _resolve_saving(self.segment_saving)
        check_interval_scorer(
            segment_saving,
            ensure_penalised=True,
            caller_name=self.__class__.__name__,
            arg_name="segment_saving",
        )
        self.segment_saving_ = clone(segment_saving).fit(X, y)

        point_saving = _resolve_saving(self.point_saving)
        check_interval_scorer(
            point_saving,
            ensure_penalised=True,
            caller_name=self.__class__.__name__,
            arg_name="point_saving",
        )
        self.point_saving_ = clone(point_saving).fit(X, y)
        if self.point_saving_.min_size > 1:
            raise ValueError(
                f"`point_saving` must have min_size == 1, "
                f"got min_size={self.point_saving_.min_size}."
            )

        self._min_segment_length = max(
            self.min_segment_length, self.segment_saving_.min_size
        )
        if self._min_segment_length > self.max_segment_length:
            raise ValueError(
                f"Effective min_segment_length ({self._min_segment_length}) must not "
                f"exceed max_segment_length ({self.max_segment_length})."
            )

        return self

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
            ``"scores"`` : np.ndarray of shape (n_samples,)
                Cumulative optimal savings from the dynamic programme.
            ``"segment_anomaly_features"`` : list of np.ndarray
                One array per segment anomaly. Each array holds 0-based feature
                indices identified as changed, ordered from strongest to weakest
                evidence. Empty arrays when the segment saving is not a
                :class:`PenalisedScore` with per-feature scores.
            ``"point_anomaly_features"`` : list of np.ndarray
                Same as ``"segment_anomaly_features"``, but for point anomalies.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True)

        segment_cache = self.segment_saving_.precompute(X)
        point_cache = self.point_saving_.precompute(X)

        opt_savings, opt_anomaly_starts = _run_capa(
            self.segment_saving_,
            segment_cache,
            self.point_saving_,
            point_cache,
            X.shape[0],
            self._min_segment_length,
            self.max_segment_length,
        )
        segment_anomalies, point_anomalies = _extract_anomalies(opt_anomaly_starts)

        # Changed features computed separately for segments and points.
        # Points are expanded to [t, t+1) intervals for evaluation.
        segment_anomaly_features = _get_changed_features(
            self.segment_saving_, segment_cache, segment_anomalies
        )
        point_intervals = (
            np.column_stack([point_anomalies, point_anomalies + 1])
            if len(point_anomalies)
            else np.empty((0, 2), dtype=np.intp)
        )
        point_anomaly_features = _get_changed_features(
            self.point_saving_, point_cache, point_intervals
        )

        return {
            "segment_anomalies": segment_anomalies,
            "point_anomalies": point_anomalies,
            "scores": opt_savings,
            "segment_anomaly_features": segment_anomaly_features,
            "point_anomaly_features": point_anomaly_features,
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
        result = self.predict_all(X)
        segment_anomalies = result["segment_anomalies"]
        if not self.include_point_anomalies or len(result["point_anomalies"]) == 0:
            return segment_anomalies
        point_intervals = np.column_stack(
            [result["point_anomalies"], result["point_anomalies"] + 1]
        )
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
        result = self.predict_all(X)
        anomalies = result["segment_anomalies"]
        if self.include_point_anomalies and len(result["point_anomalies"]):
            point_intervals = np.column_stack(
                [result["point_anomalies"], result["point_anomalies"] + 1]
            )
            anomalies = (
                np.vstack([anomalies, point_intervals])
                if len(anomalies)
                else point_intervals
            )
        if len(anomalies) == 0:
            return np.empty(0, dtype=np.intp)
        n_samples = len(result["scores"])
        boundaries = np.unique(anomalies)
        return boundaries[(boundaries > 0) & (boundaries < n_samples)].astype(np.intp)

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
        result = self.predict_all(X)
        n_samples = len(result["scores"])
        labels = np.zeros(n_samples, dtype=np.intp)

        intervals = list(result["segment_anomalies"])
        if self.include_point_anomalies:
            intervals += [[t, t + 1] for t in result["point_anomalies"]]
        intervals.sort(key=lambda x: x[0])

        for label, (start, end) in enumerate(intervals, start=1):
            labels[start:end] = label
        return labels
