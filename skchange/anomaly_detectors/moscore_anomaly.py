"""The Moving Score algorithm for multiple collective anomaly detection."""

__author__ = ["mtveten"]
__all__ = ["MoscoreAnomaly"]

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from skchange.anomaly_detectors.base import CollectiveAnomalyDetector
from skchange.anomaly_detectors.circular_binseg import greedy_anomaly_selection
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
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
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


class MoscoreAnomaly(CollectiveAnomalyDetector):
    """Moving score algorithm for multiple collective anomaly detection.

    A custom version of the MOSUM (moving sum) algorithm [1]_ for collective anomaly
    detection. It runs a test statistic for a single anomaly of user-specified lengths
    across all the data, and compares the values in the anomaly window with
    `left_bandwidth` values to the left and `right_bandwidth` samples to the right of
    the anomaly window.

    Experimental.

    Efficently implemented using numba.

    Parameters
    ----------
    score: str, tuple[Callable, Callable], optional (default="mean")
        Test statistic to use for changepoint detection.
        * If "mean", the difference-in-mean statistic is used,
        * If "var", the difference-in-variance statistic is used,
        * If a tuple, it must contain two functions: The first function is the scoring
        function, which takes in the output of the second function as its first
        argument, and start, end and split indices as the second, third and fourth
        arguments. The second function is the initializer, which precomputes quantities
        that should be precomputed. See skchange/scores/score_factory.py for examples.
    min_anomaly_length : int, optional (default=2)
        Minimum length of a collective anomaly.
    max_anomaly_length : int, optional (default=100)
        Maximum length of a collective anomaly. Must be no larger than
        `left_bandwidth + right_bandwidth`.
    left_bandwidth : int, optional (default=50)
        Number of samples to the left of the anomaly window to use in the test
        statistic.
    right_bandwidth : int, optional (default=left_bandwidth)
        Number of samples to the right of the anomaly window to use in the test
        statistic. If None, set to `left_bandwidth`.
    threshold_scale : float, optional (default=2.0)
        Scaling factor for the threshold. The threshold is set to
        'threshold_scale * default_threshold', where the default threshold depends on
        the number of samples, the number of variables, `bandwidth` and `level`.
        If None, the threshold is tuned on the data input to .fit().
    level : float, optional (default=0.01)
        If `threshold_scale` is None, the threshold is set to the (1-`level`)-quantile
        of the anomaly score on the training data. For this to be correct, the
        training data must contain no anomalies.
    anomaly_lengths : np.ndarray, optional (default=None)
        Lengths of anomalies to consider. If None, all lengths between
        `min_anomaly_length` and `max_anomaly_length` are considered. If it is not
        important to consider all candidates, just a sparse subset for example,
        customising the anomaly lengths can significantly speed up the algorithm.

    References
    ----------
    .. [1] Eichinger, B., & Kirch, C. (2018). A MOSUM procedure for the estimation of
    multiple random change points.

    Examples
    --------
    from skchange.anomaly_detectors.moscore_anomaly import MoscoreAnomaly
    from skchange.datasets.generate import generate_teeth_data

    anomalies = [(100, 119), (250, 299)]
    df = generate_anomalous_data(500, anomalies=anomalies, means=[10.0, 5.0])
    detector = MoscoreAnomaly()
    detector.fit_predict(df)
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        score: Union[str, tuple[Callable, Callable]] = "mean",
        min_anomaly_length: int = 2,
        max_anomaly_length: int = 100,
        left_bandwidth: int = 50,
        right_bandwidth: int = None,
        threshold_scale: Optional[float] = 2.0,
        level: float = 0.01,
        anomaly_lengths: np.ndarray = None,
    ):
        self.score = score
        self.min_anomaly_length = min_anomaly_length
        self.max_anomaly_length = max_anomaly_length
        self.left_bandwidth = left_bandwidth
        self.right_bandwidth = right_bandwidth
        self.threshold_scale = threshold_scale
        self.level = level
        self.anomaly_lengths = anomaly_lengths
        super().__init__()

        self.score_f, self.score_init_f = anomaly_score_factory(score)
        self._right_bandwidth = right_bandwidth if right_bandwidth else left_bandwidth
        if anomaly_lengths is None:
            self._anomaly_lengths = np.arange(
                min_anomaly_length, max_anomaly_length + 1
            )
            self._min_anomaly_length = min_anomaly_length
            self._max_anomaly_length = max_anomaly_length
        else:
            self._anomaly_lengths = np.asarray(anomaly_lengths)
            self._min_anomaly_length = anomaly_lengths.min()
            self._max_anomaly_length = anomaly_lengths.max()

        check_larger_than(1, self._min_anomaly_length, "_min_anomaly_length")
        check_larger_than(
            self._min_anomaly_length, self._max_anomaly_length, "_max_anomaly_length"
        )
        check_smaller_than(
            self.left_bandwidth + self._right_bandwidth,
            self._max_anomaly_length,
            "_max_anomaly_length",
        )
        check_larger_than(1, self.left_bandwidth, "left_bandwidth")
        check_larger_than(1, self._right_bandwidth, "_right_bandwidth")
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
            self._anomaly_lengths,
            self.left_bandwidth,
            self._right_bandwidth,
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

    def _fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """Fit to training data.

        Sets the threshold of the detector.
        If `threshold_scale` is None, the threshold is set to the (1-`level`)-quantile
        of the change/anomaly scores on the training data. For this to be correct,
        the training data must contain no changepoints. If `threshold_scale` is a
        number, the threshold is set to `threshold_scale` times the default threshold
        for the detector. The default threshold depends at least on the data's shape,
        but could also depend on more parameters.

        Parameters
        ----------
        X : pd.DataFrame
            training data to fit the threshold to.
        y : pd.Series, optional
            Does nothing. Only here to make the fit method compatible with sktime
            and scikit-learn.

        Returns
        -------
        self : returns a reference to self
        """
        min_length = (
            self.left_bandwidth + self._right_bandwidth + self._min_anomaly_length
        )
        X = check_data(
            X,
            min_length=min_length,
            min_length_name="left_bandwidth + _right_bandwidth + _min_anomaly_length",
        )
        self.threshold_ = self._get_threshold(X)
        return self

    def _predict(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Detect events in test/deployment data.

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
        min_length = (
            self.left_bandwidth + self._right_bandwidth + self._min_anomaly_length
        )
        X = check_data(
            X,
            min_length=min_length,
            min_length_name="left_bandwidth + _right_bandwidth + _min_anomaly_length",
        )
        anomalies, scores, starts, ends = run_moscore_anomaly(
            X.values,
            self.score_f,
            self.score_init_f,
            self._anomaly_lengths,
            self.left_bandwidth,
            self._right_bandwidth,
            self.threshold_,
        )
        self.scores = pd.DataFrame(
            {"anomaly_start": starts, "anomaly_end": ends, "score": scores}
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
            {
                "score": "mean",
                "min_anomaly_length": 2,
                "max_anomaly_length": 8,
                "left_bandwidth": 4,
            },
            {
                "score": "mean",
                "min_anomaly_length": 2,
                "max_anomaly_length": 6,
                "left_bandwidth": 3,
            },
        ]
        return params
