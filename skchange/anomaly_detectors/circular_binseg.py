"""Circular binary segmentation algorithm for multiple changepoint detection."""

__author__ = ["Tveten"]
__all__ = ["CircularBinarySegmentation"]

from typing import Optional, Union

import numpy as np
import pandas as pd

from skchange.anomaly_detectors.base import CollectiveAnomalyDetector
from skchange.anomaly_scores import BaseLocalAnomalyScore, to_local_anomaly_score
from skchange.change_detectors.seeded_binseg import make_seeded_intervals
from skchange.costs import BaseCost, L2Cost
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
    X: np.ndarray,
    score: BaseLocalAnomalyScore,
    threshold: float,
    min_segment_length: int,
    max_interval_length: int,
    growth_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts, ends = make_seeded_intervals(
        X.shape[0],
        2 * min_segment_length,
        max_interval_length,
        growth_factor,
    )
    score.fit(X)

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


class CircularBinarySegmentation(CollectiveAnomalyDetector):
    """Circular binary segmentation algorithm for multiple collective anomaly detection.

    Binary segmentation type changepoint detection algorithms recursively split the data
    into two segments, and test whether the two segments are different. Circular binary
    segmentation [1]_ is a variant of binary segmentation where the statistical test
    (anomaly score) is applied to compare the data behaviour of an inner interval subset
    with the surrounding data contained in an outer interval.
    In other words, the null hypothesis within each outer interval is that the data
    is stationary, while the alternative hypothesis is that there is a collective
    anomaly within the outer interval.

    Efficently implemented using numba.

    Parameters
    ----------
    anomaly_score : BaseLocalAnomalyScore or BaseCost, optional, default=L2Cost()
        The local anomaly score to use for anomaly detection. If a cost is given, it is
        converted to a local anomaly score using the `LocalAnomalyScore` class.
    threshold_scale : float, default=2.0
        Scaling factor for the threshold. The threshold is set to
        `threshold_scale * 2 * p * np.sqrt(np.log(n))`, where `n` is the sample size
        and `p` is the number of variables. If None, the threshold is tuned on the
        data input to `.fit()`.
    level : float, default=0.01
        If `threshold_scale` is None, the threshold is set to the (1-`level`)-quantile
        of the changepoint scores of all the seeded intervals on the training data.
        For this to be correct, the training data must contain no changepoints.
    min_segment_length : int, default=5
        Minimum length between two changepoints. Must be greater than or equal to 1.
    max_interval_length : int, default=100
        The maximum length of an interval to estimate a changepoint in. Must be greater
        than or equal to `2 * min_segment_length`.
    growth_factor : float, default=1.5
        The growth factor for the seeded intervals. Intervals grow in size according to
        `interval_len=max(interval_len + 1, np.floor(growth_factor * interval_len))`,
        starting at `interval_len=min_interval_length`. It also governs the amount
        of overlap between intervals of the same length, as the start of each interval
        is shifted by a factor of `1 + 1 / growth_factor`. Must be a float in
        `(1, 2]`.

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
        anomaly_score: Optional[Union[BaseCost, BaseLocalAnomalyScore]] = None,
        threshold_scale: Optional[float] = 2.0,
        level: float = 1e-8,
        min_segment_length: int = 5,
        max_interval_length: int = 1000,
        growth_factor: float = 1.5,
    ):
        self.anomaly_score = anomaly_score
        self.threshold_scale = threshold_scale  # Just holds the input value.
        self.level = level
        self.min_segment_length = min_segment_length
        self.max_interval_length = max_interval_length
        self.growth_factor = growth_factor
        super().__init__()

        _anomaly_score = L2Cost() if anomaly_score is None else anomaly_score
        self._anomaly_score = to_local_anomaly_score(_anomaly_score)

        check_larger_than(0.0, self.threshold_scale, "threshold_scale", allow_none=True)
        check_in_interval(pd.Interval(0.0, 1.0, closed="neither"), self.level, "level")
        check_larger_than(1.0, self.min_segment_length, "min_segment_length")
        check_larger_than(
            2 * self.min_segment_length, self.max_interval_length, "max_interval_length"
        )
        check_in_interval(
            pd.Interval(1.0, 2.0, closed="right"),
            self.growth_factor,
            "growth_factor",
        )

    def _tune_threshold(self, X: pd.DataFrame) -> float:
        """Tune the threshold.

        The threshold is set to the (1-`level`)-quantile of the changepoint scores
        from all the seeded intervals on the training data `X`. For this to be
        correct, the training data must contain no changepoints.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to tune the threshold on.

        Returns
        -------
        threshold : float
            The tuned threshold.
        """
        _, scores, _, _, _ = run_circular_binseg(
            X.values,
            self._anomaly_score,
            np.inf,
            self.min_segment_length,
            self.max_interval_length,
            self.growth_factor,
        )
        return np.quantile(scores, 1 - self.level)

    @staticmethod
    def get_default_threshold(n: int, p: int, max_interval_length) -> float:
        """Get the default threshold for Circular Binary Segmentation.

        Parameters
        ----------
        n : int
            Sample size.
        p : int
            Number of variables.

        Returns
        -------
        threshold : float
            The default threshold.
        """
        return 2 * p * np.log(n * max_interval_length)

    def _get_threshold(self, X: pd.DataFrame) -> float:
        if self.threshold_scale is None:
            return self._tune_threshold(X)
        else:
            return self.threshold_scale * self.get_default_threshold(
                X.shape[0], X.shape[1], self.max_interval_length
            )

    def _fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """Fit to training data.

        Sets the threshold of the detector.
        If `threshold_scale` is None, the threshold is set to the (1-`level`)-quantile
        of the change/anomaly scores on the training data. For this to be
        correct, the training data must contain no changepoints. If `threshold_scale`
        is a number, the threshold is set to `threshold_scale` times the default
        threshold for the detector. The default threshold depends at least on the data's
        shape, but could also depend on more parameters.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit the threshold to.
        y : pd.Series, optional
            Does nothing. Only here to make the fit method compatible with `sktime`
            and `scikit-learn`.

        Returns
        -------
        self : Returns a reference to self
        """
        X = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="min_interval_length",
        )
        self.threshold_ = self._get_threshold(X)
        return self

    def _predict(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Detect events in test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Data to detect events in (time series).

        Returns
        -------
        pd.Series[pd.Interval] containing the collective anomaly intervals.

        Notes
        -----
        The start and end points of the intervals can be accessed by
        `output.array.left` and `output.array.right`, respectively.
        """
        X = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="min_interval_length",
        )
        anomalies, scores, maximizers, starts, ends = run_circular_binseg(
            X.values,
            self._anomaly_score,
            self.threshold_,
            self.min_segment_length,
            self.max_interval_length,
            self.growth_factor,
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
        return CollectiveAnomalyDetector._format_sparse_output(anomalies)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for annotators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from skchange.costs import MultivariateGaussianCost, L2Cost

        params = [
            {"anomaly_score": L2Cost(), "threshold_scale": 5},
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
