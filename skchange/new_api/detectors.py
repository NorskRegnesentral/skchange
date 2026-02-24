"""Moving window change-point detection algorithm."""

from numbers import Integral, Real

import numpy as np
from sklearn.utils.validation import check_is_fitted, validate_data

from skchange.change_detectors._moving_window import (
    make_extended_moving_window_cuts,
    select_changepoints_by_bottom_up,
    select_changepoints_by_detection_length,
    select_changepoints_by_local_optimum,
)
from skchange.new_api._utils_param_validation import (
    HasMethods,
    Interval,
    StrOptions,
    _fit_context,
)
from skchange.new_api.base import BaseChangeDetector
from skchange.new_api.scorers import CUSUM, IntervalScorer, PenalisedScore
from skchange.new_api.typing import ArrayLike, Segmentation, Self
from skchange.new_api.utils import (
    check_interval_scorer,
    check_penalty,
    make_segmentation,
    to_change_score,
)


def transform_multiple_moving_window(
    fitted_score: IntervalScorer,
    X: np.ndarray,
    bandwidths: np.ndarray,
) -> np.ndarray:
    """Compute moving-window score series for one or multiple bandwidths.

    Parameters
    ----------
    fitted_score : IntervalScorer
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
    precomputed = fitted_score.precompute(X)

    scores = np.full((n_samples, len(bandwidths)), np.nan)
    for i, bw in enumerate(bandwidths):
        cuts = make_extended_moving_window_cuts(
            n_samples,
            int(bw),
            fitted_score.min_size,
        )
        cuts_scores = fitted_score.evaluate(precomputed, cuts).reshape(-1)
        scores[cuts[:, 1], i] = cuts_scores

    return scores


class MovingWindow(BaseChangeDetector):
    """Moving window algorithm for multiple change-point detection.

    The MOSUM (moving sum) algorithm [1]_, but generalized to allow for any change
    score, both penalised and unpenalised. The basic algorithm runs a test statistic for
    a
    single change-point across the data in a moving window fashion.
    In each window, the data is split into two equal halves with `bandwidth` samples
    on either side of a split point.
    This process generates a time series of penalised scores, which are used to generate
    candidate change-points as local maxima within intervals where the penalised scores
    are all above zero.
    The final set of change-points is selected from the candidate change-points using
    one of the two selection methods described in [2]_.

    Several of the extensions available in the mosum R package [2]_ are also available
    in this implementation, including the ability to use multiple bandwidths. The
    CUSUM-type boundary extension for computing the test statistic for candidate change-
    points less than `bandwidth` samples from the start and end of the data is also
    implemented by default.

    Parameters
    ----------
    change_score : BaseIntervalScorer, optional, default=CUSUM()
        The change score to use in the algorithm. If a cost is given, it is
        converted to a change score using the `ChangeScore` class.
    penalty : np.ndarray or float, optional, default=None
        The penalty to use for change detection. If the score is
        penalised (`change_score.__sklearn_tags__().interval_scorer_tags.penalised`)
        the penalty will be ignored. The different types of penalties are as follows:

        * ``float``: A constant penalty applied to the sum of scores across all
          variables in the data.
        * ``np.ndarray``: A penalty array of the same length as the number of
          columns in the data, where element ``i`` of the array is the penalty for
          ``i+1`` variables being affected by a change. The penalty array
          must be positive and increasing (not strictly). A penalised score with a
          linear penalty array is faster to evaluate than a nonlinear penalty array.
        * ``None``: A default penalty is created in `predict` based on the fitted
          score using the `make_bic_penalty` function.

    bandwidth : int or list of int, default=None
        The bandwidth is the number of samples on either side of a candidate
        change-point. Must be 1 or greater. If ``None``, a data-dependent default
        is chosen in ``fit`` as ``max(1, min(20, n_samples // 10))``. If a list of
        bandwidths is given, the algorithm will run for each bandwidth in the list
        and combine the results accoring to the "bottom-up" merging approach
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
    >>> from skchange.change_detectors import MovingWindow
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
        "penalty": ["array-like", Real, None],
        "bandwidth": ["array-like", Interval(Integral, 1, None, closed="left"), None],
        "selection_method": [StrOptions({"local_optimum", "detection_length"})],
        "min_detection_fraction": [Interval(Real, 0, 0.5, closed="neither")],
        "local_optimum_fraction": [Interval(Real, 0, None, closed="right")],
    }

    def __init__(
        self,
        change_score: IntervalScorer | None = None,
        penalty: ArrayLike | float | None = None,
        bandwidth: ArrayLike | int | None = None,
        selection_method: str = "local_optimum",
        min_detection_fraction: float = 0.2,
        local_optimum_fraction: float = 0.4,
    ):
        self.change_score = change_score
        self.penalty = penalty
        self.bandwidth = bandwidth
        self.selection_method = selection_method
        self.min_detection_fraction = min_detection_fraction
        self.local_optimum_fraction = local_optimum_fraction

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
        X = validate_data(
            self,
            X,
            reset=True,
            ensure_2d=True,
        )

        scorer = self.change_score or CUSUM()
        scorer = check_interval_scorer(
            scorer,
            allow_penalised=True,
            clone=True,
            caller_name=self.__class__.__name__,
            arg_name="change_score",
        )
        scorer = to_change_score(
            scorer,
            caller_name=self.__class__.__name__,
            arg_name="change_score",
        )
        if not scorer.__sklearn_tags__().interval_scorer_tags.penalised:
            penalty = self.penalty or scorer.get_default_penalty(X.shape[0], X.shape[1])
            penalty = check_penalty(
                penalty, caller_name=self.__class__.__name__, arg_name="penalty"
            )
            scorer = PenalisedScore(scorer, penalty)
        self.change_score_ = scorer.fit(X, y)

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

        min_required_samples = 2 * int(np.max(self.bandwidth_))
        if X.shape[0] < min_required_samples:
            raise ValueError(
                "Insufficient samples for configured `bandwidth`: "
                f"n_samples={X.shape[0]} but at least "
                f"2 * max(bandwidth) = {min_required_samples} is required."
            )

        if self.selection_method == "detection_length" and len(self.bandwidth_) > 1:
            raise ValueError(
                'The selection method `"detection_length"` is not supported for'
                'multiple bandwidths. Use `"local_optimum"` instead.'
            )

        return self

    def predict(self, X: ArrayLike) -> Segmentation:
        """Detect changepoints in a time series.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyze for changepoints.

        Returns
        -------
        result : dict
            Detection result as a dict with fields:

            - "changepoints": np.ndarray, changepoint indices
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
            self.change_score_,
            X,
            self.bandwidth_,
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

        return make_segmentation(changepoints=changepoints)
