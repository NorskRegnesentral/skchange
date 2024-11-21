"""Seeded binary segmentation algorithm for multiple changepoint detection."""

__author__ = ["Tveten"]
__all__ = ["SeededBinarySegmentation"]

from typing import Optional, Union

import numpy as np
import pandas as pd

from skchange.change_detectors.base import ChangeDetector
from skchange.change_scores import BaseChangeScore, to_change_score
from skchange.costs import BaseCost, L2Cost
from skchange.utils.numba import njit
from skchange.utils.validation.data import check_data
from skchange.utils.validation.parameters import check_in_interval, check_larger_than


@njit
def make_seeded_intervals(
    n: int, min_length: int, max_length: int, growth_factor: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    starts = [0]  # For numba to be able to compile type.
    ends = [0]  # For numba to be able to compile type.
    step_factor = 1 - 1 / growth_factor
    max_length = min(max_length, n)
    n_lengths = int(np.ceil(np.log(max_length / min_length) / np.log(growth_factor)))
    interval_lens = np.unique(np.round(np.geomspace(min_length, max_length, n_lengths)))
    for interval_len in interval_lens:
        step = max(1, np.round(step_factor * interval_len))
        n_steps = int(np.ceil((n - interval_len) / step))
        new_starts = [int(i * step) for i in range(n_steps + 1)]
        starts += new_starts
        new_ends = [
            int(min(i * step + interval_len - 1, n - 1)) for i in range(n_steps + 1)
        ]
        ends += new_ends
        if ends[-1] - starts[-1] + 1 < min_length:
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
        scores[(cpt >= starts) & (cpt < ends)] = 0.0
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
        splits_lower = start + min_segment_length - 1
        splits = np.arange(splits_lower, end - min_segment_length + 1)
        intervals = np.column_stack(
            (np.repeat(start, splits.size), splits + 1, np.repeat(end + 1, splits.size))
        )
        scores = change_score.evaluate(intervals)
        agg_scores = np.sum(scores, axis=1)
        argmax = np.argmax(agg_scores)
        amoc_scores[i] = agg_scores[argmax]
        maximizers[i] = splits_lower + argmax

    cpts = greedy_changepoint_selection(
        amoc_scores, maximizers, starts, ends, threshold
    )
    return cpts, amoc_scores, maximizers, starts, ends


class SeededBinarySegmentation(ChangeDetector):
    """Seeded binary segmentation algorithm for multiple changepoint detection.

    Binary segmentation type changepoint detection algorithms recursively split the data
    into two segments, and test whether the two segments are different. The seeded
    binary segmentation algorithm is an efficient version of such algorithms, which
    tests for changepoints in intervals of exponentially growing length. It has the same
    theoretical guarantees as the original binary segmentation algorithm, but runs
    in log-linear time no matter the changepoint configuration.

    Efficiently implemented using numba.

    Parameters
    ----------
    change_score : BaseChangeScore or BaseCost, default=L2Cost()
        The change score to use in the algorithm. If a cost function is given, it is
        converted to a change score using the `ChangeScore` class.
    threshold_scale : float, default=2.0
        Scaling factor for the threshold. The threshold is set to
        `threshold_scale * 2 * p * np.sqrt(np.log(n))`, where `n` is the sample size
        and `p` is the number of variables. If None, the threshold is tuned on the
        data input to `fit`.
    level : float, default=0.01
        If `threshold_scale` is None, the threshold is set to the
        (1-`level`)-quantile of the changepoint scores of all the seeded intervals on
        the training data. For this to be correct, the training data must contain no
        changepoints.
    min_segment_length : int, default=5
        Minimum length between two changepoints. Must be greater than or equal to 1.
    max_interval_length : int, default=200
        The maximum length of an interval to estimate a changepoint in. Must be greater
        than or equal to `2 * min_segment_length`.
    growth_factor : float, default=1.5
        The growth factor for the seeded intervals. Intervals grow in size according to
        `interval_len=max(interval_len + 1, np.floor(growth_factor * interval_len))`,
        starting at `interval_len=min_interval_length`. It also governs the amount
        of overlap between intervals of the same length, as the start of each interval
        is shifted by a factor of `1 + 1 / growth_factor`. Must be a float in (1, 2].

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
            n_segments=4, mean=10, segment_length=100000, p=5
        )
    >>> detector = SeededBinarySegmentation()
    >>> detector.fit_predict(df)
    0     99999
    1    199999
    2    299999
    Name: changepoint, dtype: int64
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        change_score: Union[BaseChangeScore, BaseCost] = L2Cost(),
        threshold_scale: Optional[float] = 2.0,
        level: float = 1e-8,
        min_segment_length: int = 5,
        max_interval_length: int = 200,
        growth_factor: float = 1.5,
    ):
        self.change_score = change_score
        self._change_score = to_change_score(change_score)
        self.threshold_scale = threshold_scale  # Just holds the input value.
        self.level = level
        self.min_segment_length = min_segment_length
        self.max_interval_length = max_interval_length
        self.growth_factor = growth_factor
        super().__init__()

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

        The threshold is set to the (1-`level`)-quantile of the changepoint scores from
        all the seeded intervals on the training data `X`. For this to be correct, the
        training data must contain no changepoints.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to tune the threshold on.

        Returns
        -------
        threshold : float
            The tuned threshold.
        """
        _, scores, _, _, _ = run_seeded_binseg(
            X.values,
            self._change_score,
            np.inf,
            self.min_segment_length,
            self.max_interval_length,
            self.growth_factor,
        )
        return np.quantile(scores, 1 - self.level)

    @staticmethod
    def get_default_threshold(n: int, p: int) -> float:
        """Get the default threshold.

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
            Does nothing. Only here to make the fit method compatible with `sktime`
            and `scikit-learn`.

        Returns
        -------
        self : returns a reference to self
        """
        X = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="min_interval_length",
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
        y : pd.Series - annotations for sequence `X`
            exact format depends on annotation type
        """
        X = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="min_interval_length",
        )
        cpts, scores, maximizers, starts, ends = run_seeded_binseg(
            X.values,
            self._change_score,
            self.threshold_,
            self.min_segment_length,
            self.max_interval_length,
            self.growth_factor,
        )
        self.scores = pd.DataFrame(
            {"start": starts, "end": ends, "argmax_cpt": maximizers, "score": scores}
        )
        return ChangeDetector._format_sparse_output(cpts)

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
            },
            {
                "change_score": L2Cost(),
                "min_segment_length": 1,
                "max_interval_length": 20,
            },
        ]
        return params
