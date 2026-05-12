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


@njit
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


@njit
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


@njit
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


def _run_seeded_binseg(
    change_score: BaseIntervalScorer,
    X: np.ndarray,
    min_split_size: int,
    max_interval_length: int,
    growth_factor: float,
    selection_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the seeded binary segmentation algorithm.

    Parameters
    ----------
    change_score : BaseIntervalScorer
        A fitted, penalised change score.
    X : np.ndarray of shape (n_samples, n_features)
        Input data.
    min_split_size : int
        Minimum number of samples on each side of a candidate split. Must be at
        least ``change_score.min_size``.
    max_interval_length : int
        Maximum length of an interval to evaluate.
    growth_factor : float
        Growth factor for the seeded intervals.
    selection_method : str
        One of ``"greedy"`` or ``"narrowest"``.

    Returns
    -------
    changepoints : np.ndarray
        Detected changepoint indices.
    max_scores : np.ndarray
        Maximum penalised score for each seeded interval.
    argmax_scores : np.ndarray
        Index of the maximum-score split for each seeded interval.
    starts : np.ndarray
        Start indices of the seeded intervals.
    ends : np.ndarray
        End indices of the seeded intervals.
    """
    check_is_fitted(change_score)
    cache = change_score.precompute(X)
    n_samples = X.shape[0]

    starts, ends = make_seeded_intervals(
        n_samples,
        2 * min_split_size,
        max_interval_length,
        growth_factor,
    )

    max_scores = np.zeros(starts.size)
    argmax_scores = np.zeros(starts.size, dtype=np.int64)

    for i, (start, end) in enumerate(zip(starts, ends)):
        splits = np.arange(start + min_split_size, end - min_split_size + 1)
        interval_specs = np.column_stack(
            (np.repeat(start, splits.size), splits, np.repeat(end, splits.size))
        )
        scores = change_score.evaluate(cache, interval_specs).reshape(-1)
        argmax = int(np.argmax(scores))
        max_scores[i] = scores[argmax]
        argmax_scores[i] = splits[argmax]

    if selection_method == "greedy":
        cpts = greedy_selection(max_scores, argmax_scores, starts, ends)
    else:  # "narrowest"
        cpts = narrowest_selection(max_scores, argmax_scores, starts, ends)

    return np.array(cpts, dtype=np.intp), max_scores, argmax_scores, starts, ends


def _resolve_change_score(
    change_score: BaseIntervalScorer | None,
) -> BaseIntervalScorer:
    """Return change_score or the default PenalisedScore(CUSUM()).

    Needed since default resolution needs to be done in both fit and __sklearn_tags__
    to ensure correct input tags are propagated.
    """
    return change_score if change_score is not None else PenalisedScore(CUSUM())


class SeededBinarySegmentation(BaseChangeDetector):
    """Seeded binary segmentation algorithm for multiple changepoint detection.

    Binary segmentation type changepoint detection algorithms recursively split the data
    into two segments and test whether the two segments are different. The seeded binary
    segmentation algorithm is an efficient version of such algorithms that tests for
    changepoints in intervals of exponentially growing length. It has the same
    theoretical guarantees as the original binary segmentation algorithm but runs in
    log-linear time regardless of the changepoint configuration.

    Parameters
    ----------
    change_score : BaseIntervalScorer or None, default=None
        A penalised change score to use in the algorithm. Must be an instance of
        ``BaseIntervalScorer`` with ``interval_scorer_tags.penalised=True``. If
        ``None``, defaults to ``PenalisedScore(CUSUM())``.

        Use :class:`PenalisedScore` to wrap any unpenalised change score or cost:

        * ``PenalisedScore(CUSUM())`` -- CUSUM with default BIC penalty
        * ``PenalisedScore(CostChangeScore(L2Cost()), penalty=5.0)`` -- change score
         based on L2 cost with fixed penalty
    min_split_size : int, default=5
        Minimum number of samples on each side of a candidate split point within
        each evaluated interval. The effective minimum used is
        ``max(min_split_size, change_score.min_size)``, so the actual minimum may be
        larger than the value provided here when the change score requires more
        samples per segment. Note that this does not impose a lower bound on the
        spacing between detected changepoints.
    max_interval_length : int or None, default=None
        The maximum length of an interval to evaluate a changepoint in. Must be at
        least ``2 * min_split_size``. If ``None``, defaults to
        ``min(200, n_samples)`` after fitting.
    growth_factor : float, default=1.5
        Growth factor for the seeded intervals. Intervals grow in size according to
        ``interval_len = max(interval_len + 1, floor(growth_factor * interval_len))``,
        starting at ``interval_len = 2 * min_split_size``. It also governs the amount
        of overlap between intervals of the same length, as the start of each interval
        is shifted by a factor of ``1 - 1 / growth_factor``. Larger values produce
        fewer, less-overlapping intervals (faster but coarser); smaller values produce
        more, more-overlapping intervals (slower but finer). Must be in ``(1, 2]``.
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
        Fitted penalised change score.
    min_split_size_ : int
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

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.detectors import SeededBinarySegmentation
    >>> rng = np.random.default_rng(2)
    >>> X = np.concatenate([rng.normal(0, 1, (100, 1)),
    ...                     rng.normal(10, 1, (100, 1))])
    >>> detector = SeededBinarySegmentation()
    >>> detector.fit(X).predict_changepoints(X)
    array([100])
    """

    _parameter_constraints = {
        "change_score": [HasMethods(["fit", "evaluate"]), None],
        "min_split_size": [Interval(Integral, 1, None, closed="left")],
        "max_interval_length": [Interval(Integral, 2, None, closed="left"), None],
        "growth_factor": [Interval(Real, 1.0, 2.0, closed="right")],
        "selection_method": [StrOptions({"greedy", "narrowest"})],
    }

    def __init__(
        self,
        change_score: BaseIntervalScorer | None = None,
        min_split_size: int = 5,
        max_interval_length: int | None = None,
        growth_factor: float = 1.5,
        selection_method: str = "greedy",
    ):
        self.change_score = change_score
        self.min_split_size = min_split_size
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

        scorer = _resolve_change_score(self.change_score)
        check_interval_scorer(
            scorer,
            ensure_penalised=True,
            caller_name=self.__class__.__name__,
            arg_name="change_score",
        )
        self.change_score_ = clone(scorer).fit(X, y)

        self.min_split_size_ = max(self.min_split_size, self.change_score_.min_size)

        if self.n_samples_in_ < 2 * self.min_split_size_:
            raise ValueError(
                f"`SeededBinarySegmentation` requires at least 2 * min_split_size "
                f"(={2 * self.min_split_size_}) samples to fit, got "
                f"n_samples={self.n_samples_in_}."
            )

        if self.max_interval_length is None:
            self.max_interval_length_ = min(200, self.n_samples_in_)
        else:
            self.max_interval_length_ = self.max_interval_length

        if self.max_interval_length_ < 2 * self.min_split_size_:
            raise ValueError(
                f"`max_interval_length` (={self.max_interval_length_}) must be at "
                f"least 2 * min_split_size (={2 * self.min_split_size_})."
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
        X = validate_data(self, X, reset=False, ensure_2d=True)

        cpts, max_scores, argmax_scores, starts, ends = _run_seeded_binseg(
            self.change_score_,
            X,
            self.min_split_size_,
            self.max_interval_length_,
            self.growth_factor,
            self.selection_method,
        )
        return {
            "changepoints": cpts,
            "interval_starts": starts,
            "interval_ends": ends,
            "interval_max_scores": max_scores,
            "interval_argmax_splits": argmax_scores,
        }

    def predict_changepoints(self, X: ArrayLike) -> np.ndarray:
        """Detect changepoints in a time series.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyse for changepoints.

        Returns
        -------
        changepoints : np.ndarray of shape (n_changepoints,)
            Sorted integer indices of detected changepoints. A changepoint at
            index ``t`` means sample ``t`` is the first sample of a new segment,
            i.e. a structural break occurs between samples ``t-1`` and ``t``.
            Empty array if no changepoints are detected.
        """
        return self.predict_all(X)["changepoints"]
