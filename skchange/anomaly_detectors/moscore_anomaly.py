"""The Moving Score algorithm for multiple collective anomaly detection."""

__author__ = ["mtveten"]
__all__ = ["MoscoreAnomaly"]

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sktime.annotation.base import BaseSeriesAnnotator

from skchange.anomaly_detectors.circular_binseg import greedy_anomaly_selection
from skchange.anomaly_detectors.utils import format_anomaly_output
from skchange.scores.score_factory import anomaly_score_factory
from skchange.utils.validation.data import check_data
from skchange.utils.validation.parameters import check_larger_than, check_smaller_than


def run_moscore_anomaly(
    X: np.ndarray,
    score_f: Callable,
    score_init_f: Callable,
    anomaly_lengths: np.ndarray,
    left_bandwidth: int,
    right_bandwidth: int,
    threshold: float,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    starts = tuple(
        np.arange(left_bandwidth, n - k - right_bandwidth)
        for k in anomaly_lengths
        if left_bandwidth + k + right_bandwidth <= n
    )
    ends = tuple(
        start + k - 1
        for start, k in zip(starts, anomaly_lengths)
        if left_bandwidth + k + right_bandwidth <= n
    )
    starts = np.concatenate(starts)
    ends = np.concatenate(ends)
    background_starts = starts - left_bandwidth
    background_ends = ends + right_bandwidth
    params = score_init_f(X)
    scores = score_f(params, background_starts, background_ends, starts, ends)
    anomalies = greedy_anomaly_selection(
        scores,
        starts,
        ends,
        background_starts,
        background_ends,
        threshold,
    )
    return anomalies, scores, starts, ends


class MoscoreAnomaly(BaseSeriesAnnotator):
    """Moving score algorithm for multiple collective anomaly detection.

    A generalized version of the MOSUM (moving sum) algorithm [1]_ for changepoint
    detection. It runs a test statistic for a single anomaly of user-specified lengths
    across all the data, and compared the values in the anomaly window with
    `left_bandwidth` values to the left and `right_bandwidth` samples to the right of
    the anomaly window.

    Efficently implemented using numba.

    Parameters
    ----------
    score: str, Tuple[Callable, Callable], optional (default="mean")
        Test statistic to use for changepoint detection.
        * If "mean", the difference-in-mean statistic is used,
        * If "var", the difference-in-variance statistic is used,
        * If a tuple, it must contain two functions: The first function is the scoring
        function, which takes in the output of the second function as its first
        argument, and start, end and split indices as the second, third and fourth
        arguments. The second function is the initializer, which precomputes quantities
        that should be precomputed. See skchange/scores/score_factory.py for examples.
    anomaly_lengths : np.ndarray (default=np.arange(5, 100))

    bandwidth : int, optional (default=30)
        The bandwidth is the number of samples on either side of a candidate
        changepoint. The minimum bandwidth depends on the
        test statistic. For "mean", the minimum bandwidth is 1.
    threshold_scale : float, optional (default=2.0)
        Scaling factor for the threshold. The threshold is set to
        'threshold_scale * default_threshold', where the default threshold depends on
        the number of samples, the number of variables, `bandwidth` and `level`.
        If None, the threshold is tuned on the data input to .fit().
    level : float, optional (default=0.01)
        If `threshold_scale` is None, the threshold is set to the (1-`level`)-quantile
        of the changepoint score on the training data. For this to be correct, the
        training data must contain no changepoints. If `threshold_scale` is a number,
        `level` is used in the default threshold, _before_ scaling.
    min_detection_interval : int, optional (default=1)
        Minimum number of consecutive scores above the threshold to be considered a
        changepoint. Must be between 1 and `bandwidth`/2.
    fmt : str {"dense", "sparse"}, optional (default="sparse")
        Annotation output format:
        * If "sparse", a sub-series of labels for only the outliers in X is returned,
        * If "dense", a series of labels for all values in X is returned.
    labels : str {"indicator", "score", "int_label"}, optional (default="int_label")
        Annotation output labels:
        * If "indicator", returned values are boolean, indicating whether a value is an
        outlier,
        * If "score", returned values are floats, giving the outlier score.
        * If "int_label", returned values are integer, indicating which segment a value
        belongs to.

    References
    ----------
    .. [1] Eichinger, B., & Kirch, C. (2018). A MOSUM procedure for the estimation of
    multiple random change points.

    Examples
    --------
    from skchange.change_detectors.moscore import Moscore
    from skchange.datasets.generate import teeth

    # Generate data
    df = teeth(n_segments=2, mean=10, segment_length=100000, p=5, random_state=2)

    # Detect changepoints
    detector = Moscore()
    detector.fit_predict(df)
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        score: Union[str, Tuple[Callable, Callable]] = "mean",
        min_anomaly_length: int = 2,
        max_anomaly_length: int = 100,
        left_bandwidth: int = 50,
        right_bandwidth: int = None,
        threshold_scale: Optional[float] = 2.0,
        level: float = 0.01,
        anomaly_lengths: np.ndarray = None,
        fmt: str = "sparse",
        labels: str = "int_label",
    ):
        self.score = score
        self.left_bandwidth = left_bandwidth
        self.right_bandwidth = right_bandwidth if right_bandwidth else left_bandwidth
        self.threshold_scale = threshold_scale
        self.level = level
        super().__init__(fmt=fmt, labels=labels)

        self.score_f, self.score_init_f = anomaly_score_factory(score)
        if anomaly_lengths is None:
            self.anomaly_lengths = np.arange(min_anomaly_length, max_anomaly_length + 1)
            self.min_anomaly_length = min_anomaly_length
            self.max_anomaly_length = max_anomaly_length
        else:
            self.anomaly_lengths = np.asarray(self.anomaly_lengths)
            self.min_anomaly_length = self.anomaly_lengths.min()
            self.max_anomaly_length = self.anomaly_lengths.max()

        check_larger_than(2, self.min_anomaly_length, "min_anomaly_length")
        check_larger_than(
            self.min_anomaly_length, self.max_anomaly_length, "max_anomaly_length"
        )
        check_smaller_than(
            self.left_bandwidth + self.right_bandwidth,
            self.max_anomaly_length,
            "max_anomaly_length",
        )
        check_larger_than(1, self.left_bandwidth, "left_bandwidth")
        check_larger_than(1, self.right_bandwidth, "right_bandwidth")
        check_larger_than(0, threshold_scale, "threshold_scale", allow_none=True)
        check_larger_than(0, self.level, "level")

    def _tune_threshold(self, X: pd.DataFrame) -> float:
        """Tune the threshold for the Moscore algorithm.

        The threshold is set to the (1-`level`)-quantile of the score on the training
        data `X`. For this to be correct, the training data must contain no
        changepoints.

        TODO: Find the threshold given an input number `k` of "permitted" changepoints
        in the training data. This can be achieved by filtering out the top `k` peaks
        of the score.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to tune the threshold on.
        """
        _, scores, _, _ = run_moscore_anomaly(
            X.values,
            self.score_f,
            self.score_init_f,
            self.anomaly_lengths,
            self.left_bandwidth,
            self.right_bandwidth,
            np.inf,
        )
        tuned_threshold = np.quantile(scores, 1 - self.level)
        return tuned_threshold

    @staticmethod
    def get_default_threshold(n: int, p: int) -> float:
        """Get the default threshold for the MoscoreAnomaly algorithm.

        It is the asymptotic critical value of the univariate 'mean' test statitic,
        multiplied by `p` to account for the multivariate case.

        Parameters
        ----------
        n : int
            Sample size.
        p : int
            Number of variables.
        bandwidth : int
            Bandwidth of the Moscore algorithm.
        level : float, optional (default=0.01)
            Significance level for the test statistic.

        Returns
        -------
        threshold : float
            Threshold for the Moscore algorithm.
        """
        return 2 * p * np.sqrt(np.log(n))

    def _get_threshold(self, X: pd.DataFrame) -> float:
        if self.threshold_scale is None:
            return self._tune_threshold(X)
        else:
            n = X.shape[0]
            p = X.shape[1]
            return self.threshold_scale * self.get_default_threshold(n, p)

    def _fit(self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None):
        """Fit to training data.

        Trains the threshold on the input data if `tune` is True. Otherwise, the
        threshold is set to the input `threshold` value if provided. If not,
        it is set to the default value for the test statistic, which depends on
        the dimension of X.

        Parameters
        ----------
        X : pd.DataFrame
            training data to fit the threshold to.
        Y : pd.Series, optional
            Does nothing. Only here to make the fit method compatible with sktime
            and scikit-learn.

        Returns
        -------
        self : returns a reference to self
        """
        min_length = (
            self.left_bandwidth + self.right_bandwidth + self.min_anomaly_length
        )
        X = check_data(
            X,
            min_length=min_length,
            min_length_name="left_bandwidth + right_bandwidth + min_anomaly_length",
        )
        self.threshold_ = self._get_threshold(X)
        return self

    def _predict(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame - data to annotate, time series

        Returns
        -------
        Y : pd.Series - annotations for sequence X
            exact format depends on annotation type
        """
        min_length = (
            self.left_bandwidth + self.right_bandwidth + self.min_anomaly_length
        )
        X = check_data(
            X,
            min_length=min_length,
            min_length_name="left_bandwidth + right_bandwidth + min_anomaly_length",
        )
        self.anomalies, scores, starts, ends = run_moscore_anomaly(
            X.values,
            self.score_f,
            self.score_init_f,
            self.anomaly_lengths,
            self.left_bandwidth,
            self.right_bandwidth,
            self.threshold_,
        )
        self.scores = pd.DataFrame(
            {"anomaly_start": starts, "anomaly_end": ends, "score": scores}
        )
        return format_anomaly_output(
            self.fmt, self.labels, X.index, self.anomalies, scores=self.scores
        )

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
            {"score": "mean"},
            {"score": "mean", "threshold_scale": 0},
        ]
        return params
