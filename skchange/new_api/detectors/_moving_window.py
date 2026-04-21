"""Moving window change-point detection algorithm."""

__author__ = ["Tveten"]

from numbers import Integral, Real

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.detectors._base import BaseChangeDetector
from skchange.new_api.interval_scorers._base import BaseIntervalScorer
from skchange.new_api.interval_scorers._change_scores.cusum import CUSUM
from skchange.new_api.interval_scorers._penalised_score import PenalisedScore
from skchange.new_api.typing import ArrayLike, Self
from skchange.new_api.utils import SkchangeTags
from skchange.new_api.utils._param_validation import (
    HasMethods,
    Interval,
    StrOptions,
    _fit_context,
)
from skchange.new_api.utils.validation import check_interval_scorer, validate_data
from skchange.utils.numba import njit
from skchange.utils.numba.general import where


@njit
def make_extended_moving_window_cuts(
    n_samples: int,
    bandwidth: int,
    min_size: int,
) -> np.ndarray:
    splits = np.arange(min_size, n_samples - min_size + 1)

    starts = splits - bandwidth
    starts[starts < 0] = 0
    max_start = n_samples - 2 * bandwidth
    starts[starts > max_start] = max_start

    ends = splits + bandwidth
    ends[ends > n_samples] = n_samples
    min_end = 2 * bandwidth
    ends[ends < min_end] = min_end

    cuts = np.column_stack((starts, splits, ends))
    return cuts


def transform_multiple_moving_window(
    fitted_score: BaseIntervalScorer,
    X: np.ndarray,
    bandwidths: np.ndarray,
) -> np.ndarray:
    """Compute moving-window score series for one or multiple bandwidths.

    Parameters
    ----------
    fitted_score : BaseIntervalScorer
        Fitted (typically penalised) score object.
    X : np.ndarray
        Input data used for evaluation.
    bandwidths : np.ndarray
        Array of bandwidth values.

    Returns
    -------
    np.ndarray
        Score matrix of shape (n_samples, n_bandwidths).
    """
    check_is_fitted(fitted_score)
    n_samples = X.shape[0]
    cache = fitted_score.precompute(X)

    scores = np.full((n_samples, len(bandwidths)), np.nan)
    for i, bw in enumerate(bandwidths):
        interval_specs = make_extended_moving_window_cuts(
            n_samples,
            int(bw),
            fitted_score.min_size,
        )
        interval_scores = fitted_score.evaluate(cache, interval_specs).reshape(-1)
        scores[interval_specs[:, 1], i] = interval_scores

    return scores


@njit
def get_candidate_changepoints(
    scores: np.ndarray,
) -> tuple[list[int], list[tuple[int, int]]]:
    detection_intervals = where(scores > 0)
    changepoints = []
    for start, end in detection_intervals:
        cpt = start + np.argmax(scores[start:end])
        changepoints.append(cpt)
    return changepoints, detection_intervals


@njit
def select_changepoints_by_detection_length(
    scores: np.ndarray, min_detection_interval: int
) -> np.ndarray:
    candidate_cpts, detection_intervals = get_candidate_changepoints(scores)
    cpts = []
    for cpt, interval in zip(candidate_cpts, detection_intervals):
        if interval[1] - interval[0] >= min_detection_interval:
            cpts.append(cpt)
    if not cpts:
        return np.zeros(0, dtype=np.int64)
    return np.array(cpts)


@njit
def select_changepoints_by_local_optimum(
    scores: np.ndarray, selection_bandwidth: int
) -> np.ndarray:
    candidate_cpts, _ = get_candidate_changepoints(scores)
    cpts = []
    for cpt in candidate_cpts:
        if np.isclose(
            scores[cpt],
            np.max(
                scores[
                    max(cpt - selection_bandwidth, 0) : cpt + selection_bandwidth + 1
                ]
            ),
        ):
            cpts.append(cpt)
    if not cpts:
        return np.zeros(0, dtype=np.int64)
    return np.array(cpts)


@njit
def select_changepoints_by_bottom_up(
    scores: np.ndarray, bandwidths: np.ndarray, local_optimum_fraction: float
) -> np.ndarray:
    # bandwidths is expected to be sorted ascending; scores[:, i] matches bandwidths[i]
    candidate_cpts = []
    for i, bw in enumerate(bandwidths):
        local_optimum_bandwidth = int(local_optimum_fraction * bw)
        candidate_cpts_bw = select_changepoints_by_local_optimum(
            scores[:, i], local_optimum_bandwidth
        )
        for candidate_cpt in candidate_cpts_bw:
            candidate_cpts.append((candidate_cpt, bw))

    if not candidate_cpts:
        return np.zeros(0, dtype=np.int64)

    cpts = [candidate_cpts[0][0]]
    for candidate_cpt, bw in candidate_cpts[1:]:
        distance_to_closest = np.min(np.abs(candidate_cpt - np.array(cpts)))
        min_distance = max(1, int(local_optimum_fraction * bw))
        if distance_to_closest >= min_distance:
            cpts.append(candidate_cpt)

    return np.sort(np.array(cpts))


def _resolve_change_score(
    change_score: BaseIntervalScorer | None,
) -> BaseIntervalScorer:
    """Return change_score or the default PenalisedScore(CUSUM()).

    Needed since default resolution need to be done in both fit and __sklearn_tags__ to
    ensure correct input tags are propagated.
    """
    return change_score if change_score is not None else PenalisedScore(CUSUM())


class MovingWindow(BaseChangeDetector):
    """Moving window algorithm for multiple change-point detection.

    The MOSUM (moving sum) algorithm [1]_, but generalized to allow for any penalised
    change score. The basic algorithm runs a test statistic for a single change-point
    across the data in a moving window fashion.
    In each window, the data is split into two equal halves with `bandwidth` samples
    on either side of a candidate change-point.
    This process generates a time series of penalised scores, which are used to generate
    candidate change-points as local maxima within intervals where the penalised scores
    are all above zero.
    The final set of change-points is selected from the candidate change-points using
    one of the selection methods described in [2]_.

    Several of the extensions available in the mosum R package [2]_ are also available
    in this implementation, including the ability to use multiple bandwidths. The
    CUSUM-type boundary extension for computing the test statistic for candidate change-
    points less than `bandwidth` samples from the start and end of the data is also
    implemented by default.

    Parameters
    ----------
    change_score : BaseIntervalScorer, optional, default=PenalisedScore(CUSUM())
        A penalised change score to use in the algorithm. Must be an instance of
        ``BaseIntervalScorer`` with ``interval_scorer_tags.penalised=True``. The
        score is evaluated over moving windows and thresholded at zero to identify
        candidate change-points.

        Use :class:`PenalisedScore` to wrap any unpenalised change score or cost:

        * ``PenalisedScore(CUSUM())`` — CUSUM with default BIC penalty
        * ``PenalisedScore(CostChangeScore(L2Cost()), penalty=5.0)`` — L2 cost with
          fixed penalty
    bandwidth : int, list of int, or None, default=None
        The number of samples on either side of a candidate change-point. Must be
        1 or greater. If ``None``, bandwidths are set automatically at ``fit`` time:

        * ``"local_optimum"`` (default): up to 5 exponentially spaced bandwidths
          between ``max(scorer.min_size, 5)`` and
          ``max(scorer.min_size, n_samples // 5)``.
        * ``"detection_length"``: a single bandwidth of
          ``max(scorer.min_size, min(50, n_samples // 5))``.

        User-supplied values must all be ``>= scorer.min_size`` (checked at fit).
        Multiple bandwidths use the bottom-up merging approach of [2]_.
    selection_method : str, default="local_optimum"
        The method used to select the final set of change-points from a set of candidate
        change-points. The options are:

        * ``"detection_length"``: Accepts a candidate change-point if the
          ``min_detection_fraction * bandwidth`` consecutive penalised scores are above
          zero. Corresponds to the epsilon-criterion in [2]_. This method is only
          available for a single bandwidth.
        * ``"local_optimum"``: Accepts a candidate change-point if it is the local
          maximum in the scores within a neighbourhood of size
          ``local_optimum_fraction * bandwidth``. Corresponds to the eta-criterion
          in [2]_. This method is used within the "bottom-up" merging approach if
          multiple bandwidths are given.
    min_detection_fraction : float, default=0.2
        Minimum fraction of the bandwidth that must have consecutive positive
        penalised scores for a candidate to be accepted. Only used with
        ``selection_method="detection_length"``. Must be in ``(0, 0.5)``.
    local_optimum_fraction : float, default=0.8
        Neighbourhood size (as a fraction of bandwidth) for the local-optimum
        criterion. A candidate is accepted if it is the maximum within
        ``local_optimum_fraction * bandwidth`` samples of itself. Only used
        with ``selection_method="local_optimum"``. Must be ``>= 0``.

    References
    ----------
    .. [1] Eichinger, B., & Kirch, C. (2018). A MOSUM procedure for the estimation of
       multiple random change points.

    .. [2] Meier, A., Kirch, C., & Cho, H. (2021). mosum: A package for moving sums in
       change-point analysis. Journal of Statistical Software, 97, 1-42.

    Examples
    --------
    >>> from skchange.new_api.detectors import MovingWindow
    >>> from skchange.new_api.interval_scorers import PenalisedScore, CUSUM
    >>> from skchange.datasets import generate_alternating_data
    >>> df = generate_alternating_data(n_segments=4, mean=10, segment_length=100, p=5)
    >>> detector = MovingWindow()
    >>> detector.fit_predict(df)
       ilocs
    0    100
    1    200
    2    300
    """

    _parameter_constraints = {
        "change_score": [HasMethods(["fit", "evaluate"]), None],
        "bandwidth": ["array-like", Interval(Integral, 1, None, closed="left"), None],
        "selection_method": [StrOptions({"local_optimum", "detection_length"})],
        "min_detection_fraction": [Interval(Real, 0, 0.5, closed="neither")],
        "local_optimum_fraction": [Interval(Real, 0, None, closed="right")],
    }

    def __init__(
        self,
        change_score: BaseIntervalScorer | None = None,
        bandwidth: ArrayLike | int | None = None,
        selection_method: str = "local_optimum",
        min_detection_fraction: float = 0.2,
        local_optimum_fraction: float = 0.8,
    ):
        self.change_score = change_score
        self.bandwidth = bandwidth
        self.selection_method = selection_method
        self.min_detection_fraction = min_detection_fraction
        self.local_optimum_fraction = local_optimum_fraction

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
        """Fit the detector to training data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training time series data.

        y : ArrayLike | None, default=None
            Training labels. Exists for sklearn API compatibility (e.g. pipelines).
            This detector is unsupervised on single series, so it is ignored.

        Returns
        -------
        self
            Fitted detector instance.
        """
        X = validate_data(self, X, reset=True, ensure_2d=True)

        scorer = _resolve_change_score(self.change_score)
        check_interval_scorer(
            scorer,
            ensure_penalised=True,
            caller_name=self.__class__.__name__,
            arg_name="change_score",
        )
        self.change_score_ = clone(scorer).fit(X, y)

        min_size = self.change_score_.min_size
        if X.shape[0] < 2 * min_size:
            raise ValueError(
                f"`MovingWindow` requires at least 2 * change_score_.min_size = "
                f"{2 * min_size} samples to fit, got n_samples={X.shape[0]}."
            )

        if self.bandwidth is None:
            min_bw = max(min_size, 5)
            max_bw = max(min_bw, X.shape[0] // 5)
            if self.selection_method == "detection_length" or max_bw == min_bw:
                bw = np.array([min(50, max_bw)], dtype=int)
            else:
                bw = np.unique(np.round(np.geomspace(min_bw, max_bw, 5)).astype(int))
        else:
            bw = np.asarray(self.bandwidth)
            if bw.ndim == 0:
                bw = bw.reshape(1)
            if bw.size == 0:
                raise ValueError("`bandwidth` must be non-empty.")
            if np.any(bw < min_size):
                raise ValueError(
                    f"All elements of `bandwidth` must be >= the scorer's "
                    f"`min_size` ({min_size}). Got bandwidth={bw.tolist()}."
                )
        # Sort so that bandwidth_[i] always corresponds to scores[:, i] in predict,
        # ensuring the bottom-up merging traverses bandwidths in ascending order.
        self.bandwidth_ = np.sort(bw.astype(int, copy=False))

        if self.selection_method == "detection_length" and len(self.bandwidth_) > 1:
            raise ValueError(
                'The selection method `"detection_length"` is not supported for'
                'multiple bandwidths. Use `"local_optimum"` instead.'
            )

        return self

    def predict_all(self, X: ArrayLike) -> dict:
        """Detect changepoints and return all outputs in a single pass.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyze for changepoints.

        Returns
        -------
        result : dict with keys:

            ``"changepoints"`` : np.ndarray of shape (n_changepoints,)
                Sorted integer indices of detected changepoints.
            ``"scores"`` : np.ndarray of shape (n_samples, n_bandwidths)
                Moving-window penalised scores. The i'th column corresponds to the
                scores for self.bandwidth_[i]. NaN where the window does not fit.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True)

        active_mask = 2 * self.bandwidth_ <= X.shape[0]
        active_bws = self.bandwidth_[active_mask]

        if len(active_bws) == 0:
            raise ValueError(
                f"`MovingWindow.predict_*` requires at least "
                f"2 * min(bandwidth_) = {2 * int(np.min(self.bandwidth_))} "
                f"samples, got n_samples={X.shape[0]}."
            )
        scores = np.full((X.shape[0], len(self.bandwidth_)), np.nan)

        scores_active = transform_multiple_moving_window(
            self.change_score_, X, active_bws
        )
        scores[:, active_mask] = scores_active

        if self.selection_method == "detection_length":
            min_detection_length = int(self.min_detection_fraction * active_bws[0])
            changepoints = select_changepoints_by_detection_length(
                scores_active.reshape(-1), min_detection_length
            )
        else:
            if len(active_bws) == 1:
                local_optimum_bandwidth = int(
                    self.local_optimum_fraction * active_bws[0]
                )
                changepoints = select_changepoints_by_local_optimum(
                    scores_active.reshape(-1), local_optimum_bandwidth
                )
            else:
                changepoints = select_changepoints_by_bottom_up(
                    scores_active, active_bws, self.local_optimum_fraction
                )

        return {
            "changepoints": changepoints.astype(np.intp),
            "scores": scores,
        }

    def predict_changepoints(self, X: ArrayLike) -> np.ndarray:
        """Detect changepoints in a time series.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyze for changepoints.

        Returns
        -------
        changepoints : np.ndarray of shape (n_changepoints,)
            Sorted integer indices of detected changepoints. A changepoint at
            index ``t`` means sample ``t`` is the first sample of a new segment,
            i.e. a structural break occurs between samples ``t-1`` and ``t``.
            Empty array if no changepoints are detected.
        """
        return self.predict_all(X)["changepoints"]
