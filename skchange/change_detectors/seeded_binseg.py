"""Seeded binary segmentation algorithm for multiple changepoint detection."""

__author__ = ["Tveten"]
__all__ = ["SeededBinarySegmentation"]

from typing import Optional, Union

import numpy as np
import pandas as pd

from skchange.change_detectors import BaseChangeDetector
from skchange.change_scores import CUSUM, BaseChangeScore, to_change_score
from skchange.costs import BaseCost
from skchange.penalties import BasePenalty, BICPenalty, as_constant_penalty
from skchange.utils.numba import njit
from skchange.utils.validation.data import check_data
from skchange.utils.validation.parameters import check_in_interval, check_larger_than
from skchange.utils.validation.penalties import check_constant_penalty


@njit
def make_seeded_intervals(
    n: int, min_length: int, max_length: int, growth_factor: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    starts = [0]  # For numba to be able to compile type.
    ends = [1]  # For numba to be able to compile type.
    step_factor = 1 - 1 / growth_factor
    max_length = min(max_length, n)
    n_lengths = int(np.ceil(np.log(max_length / min_length) / np.log(growth_factor)))
    interval_lens = np.unique(np.round(np.geomspace(min_length, max_length, n_lengths)))
    for interval_len in interval_lens:
        step = max(1, np.round(step_factor * interval_len))
        n_steps = int(np.ceil((n - interval_len) / step))
        new_starts = [int(i * step) for i in range(n_steps + 1)]
        starts += new_starts
        new_ends = [int(min(i * step + interval_len, n)) for i in range(n_steps + 1)]
        ends += new_ends
        if ends[-1] - starts[-1] < min_length:
            starts[-1] = n - min_length
    return np.array(starts[1:]), np.array(ends[1:])


@njit
def greedy_changepoint_selection(
    scores: np.ndarray,
    maximizers: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    threshold: float,
) -> list[int]:
    scores = scores.copy()
    cpts = []
    while np.any(scores > threshold):
        argmax = scores.argmax()
        cpt = maximizers[argmax]
        cpts.append(int(cpt))
        # remove intervals that contain the detected changepoint.
        scores[(cpt >= starts) & (cpt <= ends - 1)] = 0.0
    cpts.sort()
    return cpts


def run_seeded_binseg(
    X: np.ndarray,
    change_score: BaseChangeScore,
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
    change_score.fit(X)

    amoc_scores = np.zeros(starts.size)
    maximizers = np.zeros(starts.size, dtype=np.int64)
    for i, (start, end) in enumerate(zip(starts, ends)):
        splits = np.arange(start + min_segment_length, end - min_segment_length + 1)
        intervals = np.column_stack(
            (np.repeat(start, splits.size), splits, np.repeat(end, splits.size))
        )
        scores = change_score.evaluate(intervals)
        agg_scores = np.sum(scores, axis=1)
        argmax = np.argmax(agg_scores)
        amoc_scores[i] = agg_scores[argmax]
        maximizers[i] = splits[0] + argmax

    cpts = greedy_changepoint_selection(
        amoc_scores, maximizers, starts, ends, threshold
    )
    return cpts, amoc_scores, maximizers, starts, ends


class SeededBinarySegmentation(BaseChangeDetector):
    """Seeded binary segmentation algorithm for multiple changepoint detection.

    Binary segmentation type changepoint detection algorithms recursively split the data
    into two segments, and test whether the two segments are different. The seeded
    binary segmentation algorithm is an efficient version of such algorithms, which
    tests for changepoints in intervals of exponentially growing length. It has the same
    theoretical guarantees as the original binary segmentation algorithm, but runs
    in log-linear time no matter the changepoint configuration.

    Parameters
    ----------
    change_score : BaseChangeScore or BaseCost, optional, default=CUSUM()
        The change score to use in the algorithm. If a cost function is given, it is
        converted to a change score using the `ChangeScore` class.
    threshold_scale : float, default=2.0
        Scaling factor for the threshold. The threshold is set to
        ``threshold_scale * 2 * p * np.sqrt(np.log(n))``, where ``n`` is the sample size
        and ``p`` is the number of variables. If ``None``, the threshold is tuned on the
        data input to `fit`.
    level : float, default=0.01
        If `threshold_scale` is ``None``, the threshold is set to the
        ``1-level`` quantile of the changepoint scores of all the seeded intervals on
        the training data. For this to be correct, the training data must contain no
        changepoints.
    min_segment_length : int, default=5
        Minimum length between two changepoints. Must be greater than or equal to 1.
    max_interval_length : int, default=200
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
    .. [1] Kovács, S., Bühlmann, P., Li, H., & Munk, A. (2023). Seeded binary
    segmentation: a general methodology for fast and optimal changepoint detection.
    Biometrika, 110(1), 249-256.

    Examples
    --------
    >>> from skchange.change_detectors import SeededBinarySegmentation
    >>> from skchange.datasets.generate import generate_alternating_data
    >>> df = generate_alternating_data(
            n_segments=4, mean=10, segment_length=100, p=5
        )
    >>> detector = SeededBinarySegmentation()
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
        penalty: Union[BasePenalty, float, None] = None,
        min_segment_length: int = 5,
        max_interval_length: int = 200,
        growth_factor: float = 1.5,
    ):
        self.change_score = change_score
        self.penalty = penalty
        self.min_segment_length = min_segment_length
        self.max_interval_length = max_interval_length
        self.growth_factor = growth_factor
        super().__init__()

        _change_score = CUSUM() if change_score is None else change_score
        self._change_score = to_change_score(_change_score)

        check_constant_penalty(self.penalty, caller=self, allow_none=True)
        check_larger_than(1.0, self.min_segment_length, "min_segment_length")
        check_larger_than(
            2 * self.min_segment_length, self.max_interval_length, "max_interval_length"
        )
        check_in_interval(
            pd.Interval(1.0, 2.0, closed="right"),
            self.growth_factor,
            "growth_factor",
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
            min_length=2 * self.min_segment_length,
            min_length_name="min_interval_length",
        )

        n = X.shape[0]
        p = X.shape[1]
        n_params = self._change_score.get_param_size(p)
        self.penalty_ = (
            BICPenalty(n, n_params)
            if self.penalty is None
            else as_constant_penalty(self.penalty)
        )

        return self

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
            * ``"ilocs"`` - integer locations of the change points.
        """
        X = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="min_interval_length",
        )
        cpts, scores, maximizers, starts, ends = run_seeded_binseg(
            X.values,
            self._change_score,
            self.penalty_.values[0],
            self.min_segment_length,
            self.max_interval_length,
            self.growth_factor,
        )
        self.scores = pd.DataFrame(
            {"start": starts, "end": ends, "argmax_cpt": maximizers, "score": scores}
        )
        return self._format_sparse_output(cpts)

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
        from skchange.costs import L2Cost

        params = [
            {
                "change_score": L2Cost(),
                "min_segment_length": 5,
                "max_interval_length": 100,
                "penalty": 30,
            },
            {
                "change_score": L2Cost(),
                "min_segment_length": 1,
                "max_interval_length": 20,
                "penalty": 10,
            },
        ]
        return params
