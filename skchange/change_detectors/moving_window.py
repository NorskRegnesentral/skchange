"""The Moving Window algorithm for multiple changepoint detection."""

__author__ = ["Tveten"]
__all__ = ["MovingWindow"]

from typing import Optional, Union

import numpy as np
import pandas as pd

from skchange.change_detectors import BaseChangeDetector
from skchange.change_scores import CUSUM, BaseChangeScore, to_change_score
from skchange.costs import BaseCost
from skchange.utils.numba import njit
from skchange.utils.numba.general import where
from skchange.utils.validation.data import check_data
from skchange.utils.validation.parameters import check_in_interval, check_larger_than


@njit
def get_moving_window_changepoints(
    scores: np.ndarray, threshold: float, min_detection_interval: int
) -> list:
    detection_intervals = where(scores > threshold)
    changepoints = []
    for interval in detection_intervals:
        start = interval[0]
        end = interval[1]
        if end - start >= min_detection_interval:
            cpt = np.argmax(scores[start:end]) + start
            changepoints.append(cpt)
    return changepoints


def moving_window_transform(
    X: np.ndarray,
    change_score: BaseChangeScore,
    bandwidth: int,
) -> tuple[list, np.ndarray]:
    change_score.fit(X)

    n = len(X)
    splits = np.arange(bandwidth, n - bandwidth + 1)
    starts = splits - bandwidth + 1
    ends = splits + bandwidth
    change_scores = change_score.evaluate(np.column_stack((starts, splits, ends)))
    agg_change_scores = np.sum(change_scores, axis=1)

    scores = np.zeros(n)
    scores[splits] = agg_change_scores
    return scores


class MovingWindow(BaseChangeDetector):
    """Moving window algorithm for multiple changepoint detection.

    A generalized version of the MOSUM (moving sum) algorithm [1]_ for changepoint
    detection. It runs a test statistic for a single changepoint at the midpoint in a
    moving window of length ``2 * bandwidth`` over the data.

    Parameters
    ----------
    change_score : BaseChangeScore or BaseCost, optional, default=`CUSUM()`
        The change score to use in the algorithm. If a cost function is given, it is
        converted to a change score using the `ChangeScore` class.
    bandwidth : int, default=30
        The bandwidth is the number of samples on either side of a candidate
        changepoint. The minimum bandwidth depends on the
        test statistic. For ``"mean"``, the minimum bandwidth is 1.
    threshold_scale : float, default=2.0
        Scaling factor for the threshold. The threshold is set to
        ``threshold_scale * default_threshold``, where the default threshold depends on
        the number of samples, the number of variables, `bandwidth` and `level`.
        If ``None``, the threshold is tuned on the input data to `fit`.
    level : float, default=0.01
        If `threshold_scale` is ``None``, the threshold is set to the
        ``1-level`` quantile of the changepoint score on the training data. For this
        to be correct, the training data must contain no changepoints. If
        `threshold_scale` is a number, `level` is used in the default threshold,
        _before_ scaling.
    min_detection_interval : int, default=1
        Minimum number of consecutive scores above the threshold to be considered a
        changepoint. Must be between ``1`` and ``bandwidth/2``.

    References
    ----------
    .. [1] Eichinger, B., & Kirch, C. (2018). A MOSUM procedure for the estimation of
    multiple random change points.

    Examples
    --------
    >>> from skchange.change_detectors import MovingWindow
    >>> from skchange.datasets.generate import generate_alternating_data
    >>> df = generate_alternating_data(
            n_segments=4, mean=10, segment_length=100, p=5
        )
    >>> detector = MovingWindow()
    >>> detector.fit_predict(df)
    0    100
    1    200
    2    300
    Name: changepoint, dtype: int64
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": ["Tveten"],
    }

    def __init__(
        self,
        change_score: Optional[Union[BaseChangeScore, BaseCost]] = None,
        bandwidth: int = 30,
        threshold_scale: Optional[float] = 2.0,
        level: float = 0.01,
        min_detection_interval: int = 1,
    ):
        self.change_score = change_score
        self.bandwidth = bandwidth
        self.threshold_scale = threshold_scale  # Just holds the input value.
        self.level = level
        self.min_detection_interval = min_detection_interval
        super().__init__()

        _change_score = CUSUM() if change_score is None else change_score
        self._change_score = to_change_score(_change_score)

        check_larger_than(1, self.bandwidth, "bandwidth")
        check_larger_than(0, threshold_scale, "threshold_scale", allow_none=True)
        check_larger_than(0, self.level, "level")
        check_in_interval(
            pd.Interval(1, max(1, self.bandwidth / 2 - 1), closed="both"),
            self.min_detection_interval,
            "min_detection_interval",
        )

    def _tune_threshold(self, X: pd.DataFrame) -> float:
        """Tune the threshold for the MovingWindow algorithm.

        The threshold is set to the ``1-level`` quantile of the score on the training
        data `X`. For this to be correct, the training data must contain no
        changepoints.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to tune the threshold on.
        """
        # TODO: Find the threshold given an input number `k` of "permitted" changepoints
        # in the training data. This can be achieved by filtering out the top `k` peaks
        # of the score.

        scores = moving_window_transform(
            X.values,
            self._change_score,
            self.bandwidth,
        )
        tuned_threshold = np.quantile(scores, 1 - self.level)
        return tuned_threshold

    @staticmethod
    def get_default_threshold(
        n: int, p: int, bandwidth: int, level: float = 0.01
    ) -> float:
        """Get the default threshold for the MovingWindow algorithm.

        It is the asymptotic critical value of the univariate 'mean' test statitic,
        multiplied by `p` to account for the multivariate case.

        Parameters
        ----------
        n : int
            Sample size.
        p : int
            Number of variables.
        bandwidth : int
            Bandwidth.
        level : float, optional (default=0.01)
            Significance level for the test statistic.

        Returns
        -------
        threshold : float
            Threshold value.
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

    def _fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """Fit to training data.

        Sets the threshold of the detector.
        If `threshold_scale` is ``None``, the threshold is set to the ``1-level``
        quantile of the change/anomaly scores on the training data. For this to be
        correct, the training data must contain no changepoints. If `threshold_scale` is
        a number, the threshold is set to `threshold_scale` times the default threshold
        for the detector. The default threshold depends at least on the data's shape,
        but could also depend on more parameters.

        In the case of the MovingWindow algorithm, the default threshold depends on the
        sample size, the number of variables, `bandwidth` and `level`.

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
            min_length=2 * self.bandwidth,
            min_length_name="2*bandwidth",
        )
        self.threshold_ = self._get_threshold(X)
        return self

    def _transform_scores(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Return scores for predicted labels on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to score (time series).

        Returns
        -------
        scores : pd.DataFrame with same index as X
            Scores for sequence `X`.
        """
        X = check_data(
            X,
            min_length=2 * self.bandwidth,
            min_length_name="2*bandwidth",
        )
        scores = moving_window_transform(
            X.values,
            self._change_score,
            self.bandwidth,
        )
        return pd.Series(scores, index=X.index, name="score")

    def _predict(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Detect events in test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series to detect change points in.

        Returns
        -------
        y_sparse : pd.DataFrame
            A `pd.DataFrame` with a range index and one column:
            * ``"ilocs"`` - integer locations of the changepoints.
        """
        self.scores = self.transform_scores(X)
        changepoints = get_moving_window_changepoints(
            self.scores.values, self.threshold_, self.min_detection_interval
        )
        return self._format_sparse_output(changepoints)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
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
        from skchange.costs import L2Cost

        params = [
            {"change_score": L2Cost(), "bandwidth": 5, "threshold_scale": 5.0},
            {"change_score": L2Cost(), "bandwidth": 5, "threshold_scale": None},
        ]
        return params
