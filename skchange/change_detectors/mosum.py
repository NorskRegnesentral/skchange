"""The MOSUM algorithm for multiple changepoint detection."""

__author__ = ["mtveten"]
__all__ = ["Mosum"]


from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit
from sktime.annotation.base import BaseSeriesAnnotator

from skchange.change_detectors.utils import format_changepoint_output
from skchange.scores.score_factory import score_factory
from skchange.utils.numba.general import where


@njit
def get_mosum_changepoints(
    mosums: np.ndarray, threshold: float, min_detection_interval: int
) -> list:
    detection_intervals = where(mosums > threshold)
    changepoints = []
    for interval in detection_intervals:
        start = interval[0]
        end = interval[1]
        if end - start + 1 >= min_detection_interval:
            cpt = np.argmax(mosums[start : end + 1]) + start
            changepoints.append(cpt)
    changepoints.append(len(mosums) - 1)  # The last index is defined as a changepoint.
    return changepoints


# Parallelizable??
@njit
def mosum_transform(
    X: np.ndarray,
    score_f: Callable,
    score_init_f: Callable,
    bandwidth: int,
) -> Tuple[list, np.ndarray]:
    params = score_init_f(X)
    n = len(X)
    mosums = np.zeros(n)
    for k in range(bandwidth - 1, n - bandwidth):
        start = k - bandwidth + 1
        end = k + bandwidth
        mosums[k] = score_f(params, start, end, k)
    return mosums


class Mosum(BaseSeriesAnnotator):
    """Moving sum (MOSUM) algorithm for multiple changepoint detection.

    An efficient implementation of the MOSUM algorithm [1]_ for changepoint detection.
    It runs a test statistic for a single changepoint at the midpoint in a moving window
    of length `2*bandwidth` over the data.

    Parameters
    ----------
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
    score: str (default="mean")
        Test statistic to use for changepoint detection.
        * If "mean", the difference-in-mean statistic is used,
        * More to come.
    bandwidth : int, optional (default=30)
        The bandwidth is the number of samples on either side of a candidate
        changepoint. The minimum bandwidth depends on the
        test statistic. For "mean", the minimum bandwidth is 1.
    threshold : float, optional (default=None)
        Threshold to use for changepoint detection.
        * If None, the threshold is set to the default value for the test statistic
        derived in [1]_.
    level : float, optional (default=0.01)
        Significance level for the test statistic. Only used in the default threshold if
        `threshold` is not provided.
    min_detection_interval : int, optional (default=1)
        Minimum number of consecutive scores above the threshold to be considered a
        changepoint. Must be between 1 and `bandwidth`/2.

    References
    ----------
    .. [1] Eichinger, B., & Kirch, C. (2018). A MOSUM procedure for the estimation of
    multiple random change points.

    Examples
    --------
    from skchange.change_detectors.mosum import Mosum
    from skchange.datasets.generate import teeth

    # Generate data
    df = teeth(n_segments=2, mean=10, segment_length=100000, p=5, random_state=2)

    # Detect changepoints
    detector = Mosum()
    detector.fit_predict(df)
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        score: str = "mean",
        bandwidth: int = 30,
        threshold: Optional[float] = None,
        level: float = 0.01,
        min_detection_interval: int = 1,
        fmt: str = "sparse",
        labels: str = "int_label",
    ):
        self.score = score
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.level = level
        self.min_detection_interval = min_detection_interval

        super().__init__(fmt=fmt, labels=labels)

        self.score_f, self.score_init_f = score_factory(self.score)

        if self.bandwidth < 1:
            raise ValueError("bandwidth must be at least 1.")
        if (
            self.min_detection_interval <= 0
            or self.min_detection_interval >= self.bandwidth / 2
        ):
            raise ValueError(
                "min_detection_interval must be between 0 and bandwidth/2."
            )

    def _fit(self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None):
        """Fit to training data.

        Does nothing. To comply with scikit-learn and sktime API.

        Parameters
        ----------
        X : pd.DataFrame
            training data to fit model to, time series
        Y : pd.Series, optional
            ground truth annotations for training if annotator is supervised

        Returns
        -------
        self : returns a reference to self
        """
        return self

    def _check_X(self, X: Union[pd.DataFrame, pd.Series]):
        if X.ndim < 2:
            X = X.to_frame()

        if X.shape[0] < 2 * self.bandwidth:
            raise ValueError(
                f"X must have at least 2*bandwidth samples (X.shape[0]={X.shape[0]}, "
                f"bandwidth={self.bandwidth})."
            )
        return X

    @staticmethod
    def get_default_threshold(n: int, p: int, bandwidth: int, level: float = 0.01):
        """Get the default threshold for the MOSUM algorithm.

        It is the asymptotic critical value of the univariate 'mean' test statitic,
        multiplied by `p` to account for the multivariate case.

        Parameters
        ----------
        n : int
            Sample size.
        p : int
            Number of variables.
        bandwidth : int
            Bandwidth of the MOSUM algorithm.
        level : float, optional (default=0.01)
            Significance level for the test statistic.

        Returns
        -------
        threshold : float
            Threshold for the MOSUM algorithm.
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
        threshold = p * (b + c) / a
        return threshold

    def _get_threshold(self, X: pd.DataFrame) -> float:
        n = X.shape[0]
        p = X.shape[1]
        threshold = (
            self.threshold
            if self.threshold
            else self.get_default_threshold(n, p, self.bandwidth, self.level)
        )
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative (threshold={threshold}).")
        return threshold

    def _predict(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Create annotations on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame - data to annotate, time series

        Returns
        -------
        Y : pd.Series - annotations for sequence X
            exact format depends on annotation type
        """
        X = self._check_X(X)
        self._threshold = self._get_threshold(X)
        self._scores = mosum_transform(
            X.values,
            self.score_f,
            self.score_init_f,
            self.bandwidth,
        )
        self._changepoints = get_mosum_changepoints(
            self._scores, self._threshold, self.min_detection_interval
        )
        return format_changepoint_output(
            self.fmt, self.labels, self._changepoints, X.index, self._scores
        )

    # todo: consider implementing this, optional
    # if not implementing, delete the _update method
    # def _update(self, X, Y=None):
    #     """Update model with new data and optional ground truth annotations.

    #     core logic

    #     Parameters
    #     ----------
    #     X : pd.DataFrame
    #         training data to update model with, time series
    #     Y : pd.Series, optional
    #         ground truth annotations for training if annotator is supervised
    #     Returns
    #     -------
    #     self : returns a reference to self

    #     State change
    #     ------------
    #     updates fitted model (attributes ending in "_")
    #     """

    # implement here
    # IMPORTANT: avoid side effects to X, fh

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
            {"score": "mean", "bandwidth": 10, "level": 0.01},
            {"score": "mean", "bandwidth": 10, "threshold": 0},
        ]
        return params
