"""Seeded binary segmentation algorithm for multiple changepoint detection."""

__author__ = ["Tveten"]
__all__ = ["SeededBinarySegmentation"]

import numpy as np
import pandas as pd

from ..base import BaseIntervalScorer
from ..change_scores import CUSUM, to_change_score
from ..compose.penalised_score import PenalisedScore
from ..penalties import BICPenalty, as_penalty
from ..penalties.base import BasePenalty
from ..utils.numba import njit
from ..utils.validation.data import check_data
from ..utils.validation.parameters import check_in_interval, check_larger_than
from .base import BaseChangeDetector


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
def greedy_selection(
    max_scores: np.ndarray,
    argmax_scores: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> list[int]:
    max_scores = max_scores.copy()
    cpts = []
    while np.any(max_scores > 0):
        argmax = max_scores.argmax()
        cpt = argmax_scores[argmax]
        cpts.append(int(cpt))
        # remove intervals that contain the detected changepoint.
        max_scores[(cpt >= starts) & (cpt < ends)] = 0.0
    cpts.sort()
    return cpts


@njit
def narrowest_selection(
    max_scores: np.ndarray,
    argmax_scores: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> list[int]:
    cpts = []
    scores_above_threshold = max_scores > 0
    candidate_starts = starts[scores_above_threshold]
    candidate_ends = ends[scores_above_threshold]
    candidate_maximizers = argmax_scores[scores_above_threshold]

    while len(candidate_starts) > 0:
        argmin = np.argmin(candidate_ends - candidate_starts)
        cpt = candidate_maximizers[argmin]
        cpts.append(int(cpt))

        # remove candidates that contain the detected changepoint.
        cpt_not_in_interval = ~((cpt >= candidate_starts) & (cpt < candidate_ends))
        candidate_starts = candidate_starts[cpt_not_in_interval]
        candidate_ends = candidate_ends[cpt_not_in_interval]
        candidate_maximizers = candidate_maximizers[cpt_not_in_interval]

    cpts.sort()
    return cpts


def run_seeded_binseg(
    penalised_score: BaseIntervalScorer,
    max_interval_length: int,
    growth_factor: float,
    selection_method: str = "greedy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    penalised_score.check_is_penalised()
    penalised_score.check_is_fitted()
    n_samples = penalised_score._X.shape[0]

    starts, ends = make_seeded_intervals(
        n_samples,
        2 * penalised_score.min_size,
        max_interval_length,
        growth_factor,
    )

    max_scores = np.zeros(starts.size)
    argmax_scores = np.zeros(starts.size, dtype=np.int64)
    for i, (start, end) in enumerate(zip(starts, ends)):
        splits = np.arange(
            start + penalised_score.min_size, end - penalised_score.min_size + 1
        )
        intervals = np.column_stack(
            (np.repeat(start, splits.size), splits, np.repeat(end, splits.size))
        )
        scores = penalised_score.evaluate(intervals)
        argmax = np.argmax(scores)
        max_scores[i] = scores[argmax]
        argmax_scores[i] = splits[0] + argmax

    if selection_method == "greedy":
        cpts = greedy_selection(max_scores, argmax_scores, starts, ends)
    elif selection_method == "narrowest":
        cpts = narrowest_selection(max_scores, argmax_scores, starts, ends)

    return cpts, max_scores, argmax_scores, starts, ends


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
    max_interval_length : int, default=200
        The maximum length of an interval to estimate a changepoint in. Must be greater
        than or equal to ``2 * change_score.min_size``.
    growth_factor : float, default=1.5
        The growth factor for the seeded intervals. Intervals grow in size according to
        ``interval_len=max(interval_len + 1, np.floor(growth_factor * interval_len))``,
        starting at ``interval_len=min_interval_length``. It also governs the amount
        of overlap between intervals of the same length, as the start of each interval
        is shifted by a factor of ``1 + 1 / growth_factor``. Must be a float in
        ``(1, 2]``.
    selection_method : str, default="greedy"
        The method to use for selecting changepoints. The options are:

        * ``"greedy"``: Selects the changepoint with the highest score, then removes all
          intervals that contain the detected changepoint. This process is repeated
          until no intervals are left with a score above the threshold.
        * ``"narrowest"``: Searches among the intervals with scores above the threshold,
          and selects the one with the narrowest interval. It then removes all
          intervals that contain the detected changepoint, and repeats these two steps
          until no intervals are left with a score above the threshold.

    References
    ----------
    .. [1] Kovács, S., Bühlmann, P., Li, H., & Munk, A. (2023). Seeded binary
    segmentation: a general methodology for fast and optimal changepoint detection.
    Biometrika, 110(1), 249-256.
    .. [2] Rafal Baranowski, Yining Chen, Piotr Fryzlewicz, Narrowest-Over-Threshold
    Detection of Multiple Change Points and Change-Point-Like Features, Journal of the
    Royal Statistical Society Series B: Statistical Methodology, Volume 81, Issue 3,
    July 2019, Pages 649-672.

    Examples
    --------
    >>> from skchange.change_detectors import SeededBinarySegmentation
    >>> from skchange.datasets import generate_alternating_data
    >>> df = generate_alternating_data(n_segments=4, mean=10, segment_length=100, p=5)
    >>> detector = SeededBinarySegmentation()
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
        max_interval_length: int = 200,
        growth_factor: float = 1.5,
        selection_method: str = "greedy",
    ):
        self.change_score = change_score
        self.penalty = penalty
        self.max_interval_length = max_interval_length
        self.growth_factor = growth_factor
        self.selection_method = selection_method
        super().__init__()

        _change_score = CUSUM() if change_score is None else change_score
        _change_score = to_change_score(_change_score)
        _penalty = as_penalty(self.penalty, default=BICPenalty())
        self._penalised_score = (
            _change_score.clone()  # need to avoid modifying the input change_score
            if _change_score.is_penalised_score
            else PenalisedScore(_change_score, _penalty)
        )

        check_in_interval(
            pd.Interval(1.0, 2.0, closed="right"),
            self.growth_factor,
            "growth_factor",
        )
        valid_selection_methods = ["greedy", "narrowest"]
        if self.selection_method not in valid_selection_methods:
            raise ValueError(
                f"Invalid selection method. Must be one of {valid_selection_methods}."
            )

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
            * ``"ilocs"`` - integer locations of the change points.

        Attributes
        ----------
        fitted_penalised_score : BaseIntervalScorer
            The fitted penalised change score used for the detection.
        """
        self.fitted_penalised_score: BaseIntervalScorer = self._penalised_score.clone()
        self.fitted_penalised_score.fit(X)
        X = check_data(
            X,
            min_length=2 * self.fitted_penalised_score.min_size,
            min_length_name="2 * fitted_change_score.min_size",
        )
        check_larger_than(
            2 * self.fitted_penalised_score.min_size,
            self.max_interval_length,
            "max_interval_length",
        )
        cpts, scores, maximizers, starts, ends = run_seeded_binseg(
            penalised_score=self.fitted_penalised_score,
            max_interval_length=self.max_interval_length,
            growth_factor=self.growth_factor,
            selection_method=self.selection_method,
        )
        self.scores = pd.DataFrame(
            {"start": starts, "end": ends, "argmax": maximizers, "max": scores}
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
                "max_interval_length": 100,
                "penalty": 30,
            },
            {
                "change_score": L2Cost(),
                "max_interval_length": 20,
                "penalty": 10,
            },
            {
                "change_score": L2Cost(),
                "max_interval_length": 20,
                "penalty": 10,
                "selection_method": "narrowest",
            },
        ]
        return params
