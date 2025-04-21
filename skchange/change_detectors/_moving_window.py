"""The Moving Window algorithm for multiple changepoint detection."""

__author__ = ["Tveten"]
__all__ = ["MovingWindow"]

import numpy as np
import pandas as pd

from ..base import BaseIntervalScorer
from ..change_scores import CUSUM, to_change_score
from ..compose.penalised_score import PenalisedScore
from ..utils.numba import njit
from ..utils.numba.general import where
from ..utils.validation.data import check_data
from ..utils.validation.interval_scorer import check_interval_scorer
from ..utils.validation.parameters import check_in_interval, check_larger_than
from ..utils.validation.penalties import check_penalty
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

    # astype(float) since penalty_ might be int, which causes all scores to be int.
    scores = np.repeat(-np.max(penalised_score.penalty_), n_samples).astype(np.float64)
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
    penalty : np.ndarray or float, optional, default=None
        The penalty to use for change detection. If
        `change_score.is_penalised_score == True` the penalty will be ignored.
        The different types of penalties are as follows:

            * ``float``: A constant penalty applied to the sum of scores across all
            variables in the data.
            * ``np.ndarray``: A penalty array of the same length as the number of
            columns in the data, where element ``i`` of the array is the penalty for
            ``i+1`` variables being affected by a change. The penalty array
            must be positive and increasing (not strictly). A penalised score with a
            linear penalty array is faster to evaluate than a nonlinear penalty array.
            * ``None``: A default penalty is created in `predict` based on the fitted
            score using the `make_bic_penalty` function.
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
        penalty: np.ndarray | float | None = None,
        bandwidth: int = 30,
        min_detection_interval: int = 1,
    ):
        self.change_score = change_score
        self.penalty = penalty
        self.bandwidth = bandwidth
        self.min_detection_interval = min_detection_interval
        super().__init__()

        _score = CUSUM() if change_score is None else change_score
        check_interval_scorer(
            _score,
            "change_score",
            "MovingWindow",
            required_tasks=["cost", "change_score"],
            allow_penalised=True,
        )
        _change_score = to_change_score(_score)

        check_penalty(penalty, "penalty", "MovingWindow")
        self._penalised_score = (
            _change_score.clone()  # need to avoid modifying the input change_score
            if _change_score.is_penalised_score
            else PenalisedScore(_change_score, penalty)
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

        Attributes
        ----------
        fitted_score : BaseIntervalScorer
            The fitted penalised change score used for the detection.
        """
        self.fitted_score: BaseIntervalScorer = self._penalised_score.clone()
        self.fitted_score.fit(X)
        X = check_data(
            X,
            min_length=2 * self.bandwidth,
            min_length_name="2*bandwidth",
        )
        scores = moving_window_transform(
            self.fitted_score,
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

        Attributes
        ----------
        fitted_score : BaseIntervalScorer
            The fitted penalised change score used for the detection.
        scores : pd.Series
            The detection scores obtained by the `transform_scores` method.
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
