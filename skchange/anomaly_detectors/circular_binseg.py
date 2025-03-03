"""Circular binary segmentation algorithm for multiple changepoint detection."""

__author__ = ["Tveten"]
__all__ = ["CircularBinarySegmentation"]

import numpy as np
import pandas as pd

from skchange.anomaly_detectors.base import BaseSegmentAnomalyDetector
from skchange.anomaly_scores import BaseLocalAnomalyScore, to_local_anomaly_score
from skchange.change_detectors.seeded_binseg import make_seeded_intervals
from skchange.costs import BaseCost, L2Cost
from skchange.penalties import BasePenalty, BICPenalty, as_penalty
from skchange.utils.numba import njit
from skchange.utils.validation.data import check_data
from skchange.utils.validation.parameters import check_in_interval, check_larger_than


@njit
def greedy_anomaly_selection(
    scores: np.ndarray,
    anomaly_starts: np.ndarray,
    anomaly_ends: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    threshold: float,
) -> list[tuple[int, int]]:
    scores = scores.copy()
    anomalies = []
    while np.any(scores > threshold):
        argmax = scores.argmax()
        anomaly_start = anomaly_starts[argmax]
        anomaly_end = anomaly_ends[argmax]
        anomalies.append((anomaly_start, anomaly_end))
        # remove intervals that overlap with the detected segment anomaly.
        scores[(anomaly_end > starts) & (anomaly_start < ends)] = 0.0
    anomalies.sort()
    return anomalies


@njit
def make_anomaly_intervals(
    interval_start: int, interval_end: int, min_segment_length: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    starts = []
    ends = []
    for i in range(interval_start + 1, interval_end - min_segment_length + 2):
        # TODO: Add support for anomaly_intervals starting at interval_start and ending
        # at interval_end. Currently blocked by interval evaluators requiring
        # strictly increasing interval input.
        for j in range(i + min_segment_length, interval_end):
            baseline_n = interval_end - j + i - interval_start
            if baseline_n >= min_segment_length:
                starts.append(i)
                ends.append(j)
    return np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


def run_circular_binseg(
    num_observations: int,
    score: BaseLocalAnomalyScore,
    threshold: float,
    min_segment_length: int,
    max_interval_length: int,
    growth_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    score.check_is_fitted()
    starts, ends = make_seeded_intervals(
        num_observations,
        2 * min_segment_length,
        max_interval_length,
        growth_factor,
    )

    anomaly_scores = np.zeros(starts.size)
    anomaly_starts = np.zeros(starts.size, dtype=np.int64)
    anomaly_ends = np.zeros(starts.size, dtype=np.int64)
    maximizers = np.zeros((starts.size, 2))
    for i, (start, end) in enumerate(zip(starts, ends)):
        anomaly_start_candidates, anomaly_end_candidates = make_anomaly_intervals(
            start, end, min_segment_length
        )
        intervals = np.column_stack(
            (
                np.repeat(start, anomaly_start_candidates.size),
                anomaly_start_candidates,
                anomaly_end_candidates,
                np.repeat(end, anomaly_start_candidates.size),
            )
        )
        scores = score.evaluate(intervals)
        agg_scores = np.sum(scores, axis=1)
        argmax = np.argmax(agg_scores)
        anomaly_scores[i] = agg_scores[argmax]
        anomaly_starts[i] = anomaly_start_candidates[argmax]
        anomaly_ends[i] = anomaly_end_candidates[argmax]

    anomalies = greedy_anomaly_selection(
        anomaly_scores, anomaly_starts, anomaly_ends, starts, ends, threshold
    )
    return anomalies, anomaly_scores, maximizers, starts, ends


class CircularBinarySegmentation(BaseSegmentAnomalyDetector):
    """Circular binary segmentation algorithm for multiple segment anomaly detection.

    Binary segmentation type changepoint detection algorithms recursively split the data
    into two segments, and test whether the two segments are different. Circular binary
    segmentation [1]_ is a variant of binary segmentation where the statistical test
    (anomaly score) is applied to compare the data behaviour of an inner interval subset
    with the surrounding data contained in an outer interval.
    In other words, the null hypothesis within each outer interval is that the data
    is stationary, while the alternative hypothesis is that there is a segment
    anomaly within the outer interval.

    Parameters
    ----------
    anomaly_score : BaseLocalAnomalyScore or BaseCost, optional, default=L2Cost()
        The local anomaly score to use for anomaly detection. If a cost is given, it is
        converted to a local anomaly score using the `LocalAnomalyScore` class.
    penalty : BasePenalty or float, optional, default=`BICPenalty`
        The penalty to use for the changepoint detection. If a float is given, it is
        interpreted as a constant penalty. If `None`, the penalty is set to a BIC
        penalty with ``n=X.shape[0]`` and
        ``n_params=anomaly_score.get_param_size(X.shape[1])``, where ``X`` is the input
        data to `fit`.
    min_segment_length : int, default=5
        Minimum length between two changepoints. Must be greater than or equal to 1.
    max_interval_length : int, default=100
        The maximum length of an interval to estimate a changepoint in. Must be greater
        than or equal to ``2 * min_segment_length``.
    growth_factor : float, default=1.5
        The growth factor for the seeded intervals. Intervals grow in size according to
        ``interval_len=max(interval_len + 1, np.floor(growth_factor * interval_len))``,
        starting at ``interval_len=min_interval_length``. It also governs the amount
        of overlap between intervals of the same length, as the start of each interval
        is shifted by a factor of ``1 + 1 / growth_factor``. Must be a float in
        ``(1, 2]``.

    References
    ----------
    .. [1] Olshen, A. B., Venkatraman, E. S., Lucito, R., & Wigler, M. (2004). Circular
    binary segmentation for the analysis of array-based DNA copy number data.
    Biostatistics, 5(4), 557-572.

    Examples
    --------
    >>> from skchange.anomaly_detectors import CircularBinarySegmentation
    >>> from skchange.datasets.generate import generate_alternating_data
    >>> df = generate_alternating_data(n_segments=5, mean=10, segment_length=20)
    >>> detector = CircularBinarySegmentation()
    >>> detector.fit_predict(df)
    0    [20, 40)
    1    [60, 80)
    Name: anomaly_interval, dtype: interval

    Notes
    -----
    Using costs to generate local anomaly scores will be significantly slower than using
    anomaly scores that are implemented directly. This is because the local anomaly
    score requires evaluating the cost at disjoint subsets of the data
    (before and after an anomaly), which is not a natural operation for costs
    implemented as interval evaluators.
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        anomaly_score: BaseCost | BaseLocalAnomalyScore | None = None,
        penalty: BasePenalty | float | None = None,
        min_segment_length: int = 5,
        max_interval_length: int = 1000,
        growth_factor: float = 1.5,
    ):
        self.anomaly_score = anomaly_score
        self.penalty = penalty
        self.min_segment_length = min_segment_length
        self.max_interval_length = max_interval_length
        self.growth_factor = growth_factor
        super().__init__()

        _anomaly_score = L2Cost() if anomaly_score is None else anomaly_score
        self._anomaly_score = to_local_anomaly_score(_anomaly_score)

        self._penalty = as_penalty(
            self.penalty, default=BICPenalty(), require_penalty_type="constant"
        )

        check_larger_than(1.0, self.min_segment_length, "min_segment_length")
        check_larger_than(
            2 * self.min_segment_length, self.max_interval_length, "max_interval_length"
        )
        check_in_interval(
            pd.Interval(1.0, 2.0, closed="right"),
            self.growth_factor,
            "growth_factor",
        )

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None):
        """Fit to training data.

        Sets the threshold of the detector.
        If `threshold_scale` is ``None``, the threshold is set to the ``1-level``
        quantile of the change/anomaly scores on the training data. For this to be
        correct, the training data must contain no changepoints. If `threshold_scale` is
        a number, the threshold is set to `threshold_scale` times the default threshold
        for the detector. The default threshold depends at least on the data's shape,
        but could also depend on more parameters.

        Parameters
        ----------
        X : pd.DataFrame
            training data to fit the threshold to.
        y : pd.Series, optional
            Does nothing. Only here to make the fit method compatible with `sktime`
            and `scikit-learn`.

        Returns
        -------
        self :
            Reference to self.

        State change
        ------------
        Creates fitted model that updates attributes ending in "_".
        """
        X = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="min_interval_length",
        )
        self.penalty_ = self._penalty.fit(X, self._anomaly_score)
        return self

    def _predict(self, X: pd.DataFrame | pd.Series) -> pd.Series:
        """Detect events in test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series to detect anomalies in.

        Returns
        -------
        y_sparse: pd.DataFrame
            A `pd.DataFrame` with a range index and two columns:
            * ``"ilocs"`` - left-closed ``pd.Interval``s of iloc based segments.
            * ``"labels"`` - integer labels ``1, ..., K`` for each segment anomaly.
        """
        X = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="min_interval_length",
        )

        num_observations = X.shape[0]
        self._anomaly_score.fit(X)

        anomalies, scores, maximizers, starts, ends = run_circular_binseg(
            num_observations=num_observations,
            score=self._anomaly_score,
            threshold=self.penalty_.values[0],
            min_segment_length=self.min_segment_length,
            max_interval_length=self.max_interval_length,
            growth_factor=self.growth_factor,
        )

        self.scores = pd.DataFrame(
            {
                "interval_start": starts,
                "interval_end": ends,
                "argmax_anomaly_start": maximizers[:, 0],
                "argmax_anomaly_end": maximizers[:, 1],
                "score": scores,
            }
        )
        return self._format_sparse_output(anomalies)

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for annotators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from skchange.costs import L2Cost, MultivariateGaussianCost

        params = [
            {"anomaly_score": L2Cost(), "penalty": 20},
            {
                "anomaly_score": L2Cost(),
                "min_segment_length": 3,
                "max_interval_length": 50,
            },
            {
                "anomaly_score": MultivariateGaussianCost(),
                "min_segment_length": 5,
                "max_interval_length": 20,
            },
        ]
        return params
