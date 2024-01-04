"""The Moving Score algorithm for multiple changepoint detection."""

__author__ = ["mtveten"]
__all__ = ["Moscore"]

from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit
from sktime.annotation.base import BaseSeriesAnnotator

from skchange.change_detectors.utils import format_changepoint_output
from skchange.scores.score_factory import score_factory
from skchange.utils.numba.general import where
from skchange.utils.validation.data import check_data
from skchange.utils.validation.parameters import check_in_interval, check_larger_than


@njit
def get_moscore_changepoints(
    moscores: np.ndarray, threshold: float, min_detection_interval: int
) -> list:
    detection_intervals = where(moscores > threshold)
    changepoints = []
    for interval in detection_intervals:
        start = interval[0]
        end = interval[1]
        if end - start + 1 >= min_detection_interval:
            cpt = np.argmax(moscores[start : end + 1]) + start
            changepoints.append(cpt)
    return changepoints


@njit
def moscore_transform(
    X: np.ndarray,
    score_f: Callable,
    score_init_f: Callable,
    bandwidth: int,
) -> Tuple[list, np.ndarray]:
    n = len(X)
    splits = np.arange(bandwidth - 1, n - bandwidth)
    starts = splits - bandwidth + 1
    ends = splits + bandwidth
    params = score_init_f(X)
    scores = np.zeros(n)
    scores[splits] = score_f(params, starts, ends, splits)
    return scores


class Moscore(BaseSeriesAnnotator):
    """Moving score algorithm for multiple changepoint detection.

    A generalized version of the MOSUM (moving sum) algorithm [1]_ for changepoint
    detection. It runs a test statistic for a single changepoint at the midpoint in a
    moving window of length `2*bandwidth` over the data. Efficently implemented using
    numba.

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
    from skchange.datasets.generate import generate_teeth_data

    df = generate_teeth_data(n_segments=2, mean=10, segment_length=100000, p=5)
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
        bandwidth: int = 30,
        threshold_scale: Optional[float] = 2.0,
        level: float = 0.01,
        min_detection_interval: int = 1,
        fmt: str = "sparse",
        labels: str = "int_label",
    ):
        self.score = score
        self.bandwidth = bandwidth
        self.threshold_scale = threshold_scale  # Just holds the input value.
        self.level = level
        self.min_detection_interval = min_detection_interval
        super().__init__(fmt=fmt, labels=labels)
        self.score_f, self.score_init_f = score_factory(self.score)

        check_larger_than(1, self.bandwidth, "bandwidth")
        check_larger_than(0, threshold_scale, "threshold_scale", allow_none=True)
        check_larger_than(0, self.level, "level")
        check_in_interval(
            pd.Interval(1, self.bandwidth / 2 - 1, closed="both"),
            self.min_detection_interval,
            "min_detection_interval",
        )

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
        scores = moscore_transform(
            X.values,
            self.score_f,
            self.score_init_f,
            self.bandwidth,
        )
        tuned_threshold = np.quantile(scores, 1 - self.level)
        return tuned_threshold

    @staticmethod
    def get_default_threshold(
        n: int, p: int, bandwidth: int, level: float = 0.01
    ) -> float:
        """Get the default threshold for the Moscore algorithm.

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
        u = n / bandwidth
        a = np.sqrt(2 * np.log(u))
        b = (
            2 * np.log(u)
            + 1 / 2 * np.log(np.log(u))
            + np.log(3 / 2)
            - 1 / 2 * np.log(np.pi)
        )
        c = -np.log(np.log(1 / np.sqrt(1 - level)))
        # TODO: Check if it's correct to multiply by p.
        return p * (b + c) / a

    def _get_threshold(self, X: pd.DataFrame) -> float:
        if self.threshold_scale is None:
            return self._tune_threshold(X)
        else:
            n = X.shape[0]
            p = X.shape[1]
            return self.threshold_scale * self.get_default_threshold(
                n, p, self.bandwidth, self.level
            )

    def _fit(self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None):
        """Fit to training data.

        Sets the threshold of the detector.
        If `threshold_scale` is None, the threshold is set to the (1-`level`)-quantile
        of the change/anomaly scores on the training data. For this to be correct,
        the training data must contain no changepoints. If `threshold_scale` is a
        number, the threshold is set to `threshold_scale` times the default threshold
        for the detector. The default threshold depends at least on the data's shape,
        but could also depend on more parameters.
        In the case of the Moscore algorithm, the default threshold depends on the
        sample size, the number of variables, `bandwidth` and `level`.

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

        State change
        ------------
        Creates the threshold_ attribute.
        """
        X = check_data(
            X,
            min_length=2 * self.bandwidth,
            min_length_name="2*bandwidth",
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
        X = check_data(
            X,
            min_length=2 * self.bandwidth,
            min_length_name="2*bandwidth",
        )
        scores = moscore_transform(
            X.values,
            self.score_f,
            self.score_init_f,
            self.bandwidth,
        )
        self.changepoints = get_moscore_changepoints(
            scores, self.threshold_, self.min_detection_interval
        )
        self.scores = pd.Series(scores, index=X.index, name="score")
        return format_changepoint_output(
            self.fmt, self.labels, self.changepoints, X.index, self.scores
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
            {"score": "mean", "bandwidth": 10},
        ]
        return params
