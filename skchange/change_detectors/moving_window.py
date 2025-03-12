"""The Moving Window algorithm for multiple changepoint detection."""

__author__ = ["Tveten"]
__all__ = ["MovingWindow"]

import numpy as np
import pandas as pd

from skchange.change_detectors import BaseChangeDetector
from skchange.change_scores import CUSUM, BaseChangeScore, to_change_score
from skchange.costs import BaseCost
from skchange.penalties import BasePenalty, BICPenalty, as_penalty
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
    change_score: BaseChangeScore,
    bandwidth: int,
) -> tuple[list, np.ndarray]:
    change_score.check_is_fitted()

    n_samples = change_score._X.shape[0]
    splits = np.arange(bandwidth, n_samples - bandwidth + 1)
    starts = splits - bandwidth + 1
    ends = splits + bandwidth
    change_scores = change_score.evaluate(np.column_stack((starts, splits, ends)))
    agg_change_scores = np.sum(change_scores, axis=1)

    scores = np.zeros(n_samples)
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
    penalty : BasePenalty or float, optional, default=`BICPenalty`
        The penalty to use for the changepoint detection. If a float is given, it is
        interpreted as a constant penalty. If `None`, the penalty is set to a BIC
        penalty with ``n=X.shape[0]`` and
        ``n_params=change_score.get_param_size(X.shape[1])``, where ``X`` is the input
        data to `fit`.
    bandwidth : int, default=30
        The bandwidth is the number of samples on either side of a candidate
        changepoint. The minimum bandwidth depends on the
        test statistic. For ``"mean"``, the minimum bandwidth is 1.
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
        change_score: BaseChangeScore | BaseCost | None = None,
        penalty: BasePenalty | float | None = None,
        bandwidth: int = 30,
        min_detection_interval: int = 1,
    ):
        self.change_score = change_score
        self.penalty = penalty
        self.bandwidth = bandwidth
        self.min_detection_interval = min_detection_interval
        super().__init__()

        _change_score = CUSUM() if change_score is None else change_score
        self._change_score = to_change_score(_change_score)

        self._penalty = as_penalty(
            self.penalty, default=BICPenalty(), require_penalty_type="constant"
        )

        check_larger_than(1, self.bandwidth, "bandwidth")
        check_in_interval(
            pd.Interval(1, max(1, self.bandwidth / 2 - 1), closed="both"),
            self.min_detection_interval,
            "min_detection_interval",
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

        self.penalty_: BasePenalty = self._penalty.clone()
        self.penalty_.fit(X, self._change_score)

        return self

    def _transform_scores(self, X: pd.DataFrame | pd.Series) -> pd.Series:
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
        self._change_score.fit(X)
        scores = moving_window_transform(
            self._change_score,
            self.bandwidth,
        )
        return pd.Series(scores, index=X.index, name="score")

    def _predict(self, X: pd.DataFrame | pd.Series) -> pd.Series:
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
        self.scores: pd.Series = self.transform_scores(X)
        changepoints = get_moving_window_changepoints(
            self.scores.values, self.penalty_.values[0], self.min_detection_interval
        )
        return self._format_sparse_output(changepoints)

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
        from skchange.costs import GaussianCost, L2Cost

        params = [
            {"change_score": L2Cost(), "bandwidth": 5, "penalty": 20},
            {"change_score": GaussianCost(), "bandwidth": 5, "penalty": 30},
        ]
        return params
