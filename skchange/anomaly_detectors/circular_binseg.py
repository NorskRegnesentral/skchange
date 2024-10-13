"""Circular binary segmentation algorithm for multiple changepoint detection."""

__author__ = ["Tveten"]
__all__ = ["CircularBinarySegmentation"]

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from numba import njit

from skchange.anomaly_detectors.base import CollectiveAnomalyDetector
from skchange.change_detectors.seeded_binseg import make_seeded_intervals
from skchange.scores.score_factory import anomaly_score_factory
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
        scores[(anomaly_end >= starts) & (anomaly_start < ends)] = 0.0
    anomalies.sort()
    return anomalies


@njit
def make_anomaly_intervals(
    interval_start: int, interval_end: int, min_segment_length: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    starts = []
    ends = []
    for i in range(interval_start + 1, interval_end - min_segment_length + 2):
        for j in range(i + min_segment_length - 1, interval_end + 1):
            baseline_n = interval_end - j + i - interval_start
            if baseline_n >= min_segment_length:
                starts.append(i)
                ends.append(j)
    return np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


@njit
def run_circular_binseg(
    X: np.ndarray,
    score_func: Callable,
    score_init_func: Callable,
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
    params = score_init_func(X)

    anomaly_scores = np.zeros(starts.size)
    anomaly_starts = np.zeros(starts.size, dtype=np.int64)
    anomaly_ends = np.zeros(starts.size, dtype=np.int64)
    maximizers = np.zeros((starts.size, 2))
    for i, (start, end) in enumerate(zip(starts, ends)):
        anomaly_start_candidates, anomaly_end_candidates = make_anomaly_intervals(
            start, end, min_segment_length
        )
        scores = score_func(
            params,
            np.repeat(start, anomaly_start_candidates.size),
            np.repeat(end, anomaly_start_candidates.size),
            anomaly_start_candidates,
            anomaly_end_candidates,
        )
        argmax = np.argmax(scores)
        anomaly_scores[i] = scores[argmax]
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
    segmentation is described in [1]_.

    Efficently implemented using numba.

    Parameters
    ----------
    score : {"mean", "mean_var", "mean_cov"}, tuple[Callable, Callable], default="mean"
        Test statistic to use for changepoint detection.

        * ``"mean"``: The CUSUM statistic for a change in mean (this is equivalent to a
          likelihood ratio test for a change in the mean of Gaussian data). For
          multivariate data, the sum of the CUSUM statistics for each dimension is used.
        * ``"mean_var"``: The likelihood ratio test for a change in the mean and/or
          variance of Gaussian data. For multivariate data, the sum of the likelihood
          ratio statistics for each dimension is used.
        * ``"mean_cov"``: The likelihood ratio test for a change in the mean and/or
          covariance matrix of multivariate Gaussian data.
        * If a tuple, it must contain two numba jitted functions:

            1. The first function is the scoring function, which takes four arguments:

                1. The output of the second function.
                2. Start indices of the intervals to score for a change.
                3. End indices of the intervals to score for a change.
                4. Split indices of the intervals to score for a change.

               For each start, split and end, the score should be calculated for the
               data intervals ``[start:split]`` and ``[split+1:end]``, meaning that both
               the starts and ends are inclusive, while split is included in the left
               interval.

            2. The second function is the initializer, which takes the data matrix as
               input and returns precomputed quantities that may speed up the score
               calculations. If not relevant, just return the data matrix.
    threshold_scale : float, default=2.0
        Scaling factor for the threshold. The threshold is set to
        ``threshold_scale * 2 * p * np.sqrt(np.log(n))``, where ``n`` is the sample size
        and ``p`` is the number of variables. If None, the threshold is tuned on the
        data input to ``.fit()``.
    level : float, default=0.01
        If `threshold_scale` is None, the threshold is set to the (1-`level`)-quantile
        of the changepoint scores of all the seeded intervals on the training data.
        For this to be correct, the training data must contain no changepoints.
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
    >>> from skchange.datasets.generate import generate_teeth_data
    >>> df = generate_teeth_data(n_segments=5, mean=10, segment_length=100)
    >>> detector = CircularBinarySegmentation()
    >>> detector.fit_predict(df)
    0    [100, 199]
    1    [300, 399]
    Name: anomaly_interval, dtype: interval
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        score: Union[str, tuple[Callable, Callable]] = "mean",
        threshold_scale: Optional[float] = 2.0,
        level: float = 1e-8,
        min_segment_length: int = 5,
        max_interval_length: int = 1000,
        growth_factor: float = 1.5,
    ):
        self.score = score
        self.threshold_scale = threshold_scale  # Just holds the input value.
        self.level = level
        self.min_segment_length = min_segment_length
        self.max_interval_length = max_interval_length
        self.growth_factor = growth_factor
        super().__init__()
        self.score_f, self.score_init_f = anomaly_score_factory(self.score)

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

        The threshold is set to the (1-``level``)-quantile of the changepoint scores
        from all the seeded intervals on the training data ``X``. For this to be
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
            self.score_f,
            self.score_init_f,
            np.inf,
            self.min_segment_length,
            self.max_interval_length,
            self.growth_factor,
        )
        return np.quantile(scores, 1 - self.level)

    @staticmethod
    def get_default_threshold(n: int, p: int) -> float:
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
        return 2 * p * np.sqrt(np.log(n))

    def _get_threshold(self, X: pd.DataFrame) -> float:
        if self.threshold_scale is None:
            return self._tune_threshold(X)
        else:
            n = X.shape[0]
            p = X.shape[1]
            return self.threshold_scale * self.get_default_threshold(n, p)

    def _fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """Fit to training data.

        Sets the threshold of the detector.
        If ``threshold_scale`` is None, the threshold is set to the (1-``level``)-
        quantile of the change/anomaly scores on the training data. For this to be
        correct, the training data must contain no changepoints. If ``threshold_scale``
        is a number, the threshold is set to ``threshold_scale`` times the default
        threshold for the detector. The default threshold depends at least on the data's
        shape, but could also depend on more parameters.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit the threshold to.
        y : pd.Series, optional
            Does nothing. Only here to make the fit method compatible with sktime
            and scikit-learn.

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
        output.array.left and output.array.right, respectively.
        """
        X = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="min_interval_length",
        )
        anomalies, scores, maximizers, starts, ends = run_circular_binseg(
            X.values,
            self.score_f,
            self.score_init_f,
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
        params = [
            {"score": "mean", "min_segment_length": 5, "max_interval_length": 50},
            {"score": "mean", "min_segment_length": 2, "max_interval_length": 20},
        ]
        return params
