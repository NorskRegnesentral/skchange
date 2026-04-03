"""Moving window change-point detection algorithm."""

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
) -> list:
    candidate_cpts, detection_intervals = get_candidate_changepoints(scores)
    cpts = [
        cpt
        for cpt, interval in zip(candidate_cpts, detection_intervals)
        if interval[1] - interval[0] >= min_detection_interval
    ]

    return cpts


@njit
def select_changepoints_by_local_optimum(
    scores: np.ndarray, selection_bandwidth: int
) -> list:
    candidate_cpts, _ = get_candidate_changepoints(scores)
    cpts = [
        cpt
        for cpt in candidate_cpts
        if np.isclose(
            scores[cpt],
            np.max(
                scores[
                    max(cpt - selection_bandwidth, 0) : cpt + selection_bandwidth + 1
                ]
            ),
        )
    ]

    return cpts


@njit
def select_changepoints_by_bottom_up(
    scores: np.ndarray, bandwidths: np.ndarray, local_optimum_fraction: float
) -> list:
    bandwidths = sorted(bandwidths)
    candidate_cpts = []
    for i, bw in enumerate(bandwidths):
        local_optimum_bandwidth = int(local_optimum_fraction * bw)
        candidate_cpts_bw = select_changepoints_by_local_optimum(
            scores[:, i], local_optimum_bandwidth
        )
        for candidate_cpt in candidate_cpts_bw:
            candidate_cpts.append((candidate_cpt, bw))

    if not candidate_cpts:
        return []

    cpts = [candidate_cpts[0][0]]
    for candidate_cpt, bw in candidate_cpts[1:]:
        distance_to_closest = np.min(np.abs(candidate_cpt - np.array(cpts)))
        local_optimum_bandwidth = int(local_optimum_fraction * bw)
        if distance_to_closest >= local_optimum_bandwidth:
            cpts.append(candidate_cpt)

    return cpts


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
    bandwidth : int or list of int, default=None
        The bandwidth is the number of samples on either side of a candidate
        change-point. Must be 1 or greater. If ``None``, a data-dependent default
        is chosen in ``fit`` as ``max(1, min(50, n_samples // 10))``. If a list of
        bandwidths is given, the algorithm will run for each bandwidth in the list
        and combine the results according to the "bottom-up" merging approach
        described in [2]_. A fibonacci sequence of bandwidths is recommended for
        multiple bandwidths by the authors in [2]_.
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
        The minimum size of the detection interval for a candidate change-point to be
        accepted in the ``"detection_length"`` selection method.
        be between ``0`` (exclusive) and ``1/2`` (exclusive).
    local_optimum_fraction : float, default=0.4
        The size of the neighbourhood around a candidate change-point used in the
        ``"local_optimum"`` selection method. Must be larger than or equal to ``0``.

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
        local_optimum_fraction: float = 0.4,
    ):
        self.change_score = change_score
        self.bandwidth = bandwidth
        self.selection_method = selection_method
        self.min_detection_fraction = min_detection_fraction
        self.local_optimum_fraction = local_optimum_fraction

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get tags, propagating input constraints from the wrapped scorer."""
        tags = super().__sklearn_tags__()
        tags.input_tags = (
            _resolve_change_score(self.change_score).__sklearn_tags__().input_tags
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

        if self.bandwidth is None:
            auto_bw = max(1, min(50, X.shape[0] // 10))
            bw = np.array([auto_bw], dtype=int)
        else:
            bw = np.asarray(self.bandwidth)
            if bw.ndim == 0:
                bw = bw.reshape(1)
            if bw.size == 0:
                raise ValueError("`bandwidth` must be non-empty.")
            if np.any(bw < 1):
                raise ValueError("All elements of `bandwidth` must be 1 or larger.")
        self.bandwidth_ = bw.astype(int, copy=False)

        if self.selection_method == "detection_length" and len(self.bandwidth_) > 1:
            raise ValueError(
                'The selection method `"detection_length"` is not supported for'
                'multiple bandwidths. Use `"local_optimum"` instead.'
            )

        return self

    def predict_changepoints(self, X: ArrayLike) -> np.ndarray:
        """Detect changepoints in a time series.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyze for changepoints.

        Returns
        -------
        changepoints : np.ndarray of shape (n_changepoints,)
            Indices where structural breaks occur.
        """
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            reset=False,
            ensure_2d=True,
            ensure_min_samples=2 * int(np.max(self.bandwidth_)),
        )

        scores = transform_multiple_moving_window(
            self.change_score_, X, self.bandwidth_
        )

        if self.selection_method == "detection_length":
            min_detection_length = int(self.min_detection_fraction * self.bandwidth_[0])
            changepoints = select_changepoints_by_detection_length(
                scores.reshape(-1), min_detection_length
            )
        else:
            if len(self.bandwidth_) == 1:
                local_optimum_bandwidth = int(
                    self.local_optimum_fraction * self.bandwidth_[0]
                )
                changepoints = select_changepoints_by_local_optimum(
                    scores.reshape(-1), local_optimum_bandwidth
                )
            else:
                changepoints = select_changepoints_by_bottom_up(
                    scores, self.bandwidth_, self.local_optimum_fraction
                )

        return np.array(changepoints, dtype=np.intp)
