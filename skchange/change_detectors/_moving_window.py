"""The Moving Window algorithm for multiple changepoint detection."""

__author__ = ["Tveten"]
__all__ = ["MovingWindow"]

import numpy as np
import pandas as pd

from ..base import BaseIntervalScorer
from ..change_scores import CUSUM, to_change_score
from ..compose.penalised_score import PenalisedScore
from ..penalties import BICPenalty, as_penalty
from ..penalties.base import BasePenalty
from ..utils.numba import njit
from ..utils.numba.general import where
from ..utils.validation.data import check_data
from ..utils.validation.parameters import check_in_interval, check_larger_than
from .base import BaseChangeDetector


@njit
def mosum_selection(scores: np.ndarray, min_detection_interval: int) -> list:
    detection_intervals = where(scores > 0)
    changepoints = []
    for interval in detection_intervals:
        start = interval[0]
        end = interval[1]
        if end - start >= min_detection_interval:
            cpt = np.argmax(scores[start:end]) + start
            changepoints.append(cpt)
    return changepoints


def moving_window_transform(
    penalised_score: BaseIntervalScorer,
    bandwidth: int,
) -> tuple[list, np.ndarray]:
    penalised_score.check_is_penalised()
    penalised_score.check_is_fitted()

    n_samples = penalised_score._X.shape[0]
    splits = np.arange(bandwidth, n_samples - bandwidth + 1)
    starts = splits - bandwidth + 1
    ends = splits + bandwidth
    cuts = np.column_stack((starts, splits, ends))

    scores = np.repeat(-penalised_score.penalty_.values[-1], n_samples)
    scores[splits] = penalised_score.evaluate(cuts)[:, 0]
    return scores


class MovingWindow(BaseChangeDetector):
    """Moving window algorithm for multiple changepoint detection.

    A generalized version of the MOSUM (moving sum) algorithm [1]_ for changepoint
    detection. It runs a test statistic for a single changepoint at the midpoint in a
    moving window of length ``2 * bandwidth`` over the data.

    Parameters
    ----------
    change_score : BaseIntervalScorer, optional, default=CUSUM()
        The change score to use in the algorithm. If a cost is given, it is
        converted to a change score using the `ChangeScore` class.
    penalty : BasePenalty, np.ndarray or float, optional, default=`BICPenalty`
        The penalty to use for the changepoint detection. If
        `change_score.is_penalised_score == True` the penalty will be ignored.
        The conversion of different types of penalties is as follows (see `as_penalty`):

        * ``float``: A constant penalty.
        * ``np.ndarray``: A penalty array of the same length as the number of columns in
        the data. It is converted internally to a constant, linear or nonlinear penalty
        depending on its values.
        * ``None``, the penalty is set to a BIC penalty with ``n=X.shape[0]`` and
        ``n_params=change_score.get_param_size(X.shape[1])``, where ``X`` is the input
        data to `predict`.
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
    >>> from skchange.datasets import generate_alternating_data
    >>> df = generate_alternating_data(n_segments=4, mean=10, segment_length=100, p=5)
    >>> detector = MovingWindow()
    >>> detector.fit_predict(df)
       ilocs
    0    100
    1    200
    2    300
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": ["Tveten"],
        "fit_is_empty": True,
    }

    def __init__(
        self,
        change_score: BaseIntervalScorer | None = None,
        penalty: BasePenalty | np.ndarray | float | None = None,
        bandwidth: int = 30,
        min_detection_interval: int = 1,
    ):
        self.change_score = change_score
        self.penalty = penalty
        self.bandwidth = bandwidth
        self.min_detection_interval = min_detection_interval
        super().__init__()

        _change_score = CUSUM() if change_score is None else change_score
        _change_score = to_change_score(_change_score)
        _penalty = as_penalty(self.penalty, default=BICPenalty())
        self._penalised_score = (
            _change_score.clone()  # need to avoid modifying the input change_score
            if _change_score.is_penalised_score
            else PenalisedScore(_change_score, _penalty)
        )

        check_larger_than(1, self.bandwidth, "bandwidth")
        check_in_interval(
            pd.Interval(1, max(1, self.bandwidth / 2 - 1), closed="both"),
            self.min_detection_interval,
            "min_detection_interval",
        )

        self.set_tags(distribution_type=_change_score.get_tag("distribution_type"))

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
        self._penalised_score.fit(X)
        scores = moving_window_transform(
            self._penalised_score,
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
        changepoints = mosum_selection(self.scores.values, self.min_detection_interval)
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
