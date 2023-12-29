"""Binary Segmentation type algorithms for multiple changepoint detection."""

__author__ = ["mtveten"]
__all__ = ["SeededBinarySegmentation"]


from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit
from sktime.annotation.base import BaseSeriesAnnotator

from skchange.change_detectors.utils import format_changepoint_output
from skchange.scores.score_factory import score_factory


@njit
def make_seeded_intervals(
    n: int,
    min_interval_length: int,
    max_interval_length: int,
    growth_factor: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    starts = [int(0)]  # For numba to be able to compile type.
    ends = [int(0)]  # For numba to be able to compile type.
    interval_len = min_interval_length
    max_interval_length = min(max_interval_length, n)
    step_factor = 1 - 1 / growth_factor
    while interval_len <= max_interval_length:
        level_starts = [int(0)]
        level_ends = [int(interval_len - 1)]
        step = max(1, np.floor(step_factor * interval_len))
        while level_ends[-1] < n - 1:
            start = level_starts[-1] + step
            level_starts.append(int(start))
            level_ends.append(int(min(start + interval_len - 1, n - 1)))
        interval_len = max(interval_len + 1, np.floor(growth_factor * interval_len))
        starts += level_starts
        ends += level_ends
    return np.array(starts[1:]), np.array(ends[1:])


@njit
def greedy_changepoint_selection(
    scores: np.ndarray,
    maximizers: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    threshold: float,
) -> List[int]:
    scores = scores.copy()
    cpts = []
    while np.any(scores > threshold):
        argmax = scores.argmax()
        cpt = maximizers[argmax]
        cpts.append(int(cpt))
        scores[(cpt >= starts) & (cpt < ends)] = 0.0
    cpts.sort()
    return cpts


@njit
def run_seeded_binary_segmentation(
    X: np.ndarray,
    score_func: Callable,
    score_init_func: Callable,
    threshold: float,
    min_interval_length: int,
    max_interval_length: int,
    growth_factor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts, ends = make_seeded_intervals(
        X.shape[0],
        min_interval_length,
        max_interval_length,
        growth_factor,
    )
    params = score_init_func(X)

    amoc_scores = np.zeros(starts.size)
    maximizers = np.zeros(starts.size)
    for i, (start, end) in enumerate(zip(starts, ends)):
        scores = np.zeros(end - start)
        # TODO: Add min_segment_length
        for k, split in enumerate(range(start, end)):
            scores[k] = score_func(params, start, end, split)
        argmax = np.argmax(scores)
        amoc_scores[i] = scores[argmax]
        maximizers[i] = start + argmax

    cpts = greedy_changepoint_selection(
        amoc_scores, maximizers, starts, ends, threshold
    )
    return cpts, amoc_scores, maximizers, starts, ends


class SeededBinarySegmentation(BaseSeriesAnnotator):
    """Seeded binary segmentation algorithm for multiple changepoint detection.

    Binary segmentation type changepoint detection algorithms recursively split the data
    into two segments, and test whether the two segments are different. The seeded
    binary segmentation algorithm is an efficient version of such algorithms, which
    tests for changepoints in intervals of exponentially growing length. It has the same
    theoretical guarantees as the original binary segmentation algorithm, but runs
    in log-linear time no matter the changepoint configuration.

    Efficently implemented using numba.

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
    score: str, Tuple[Callable, Callable], optional (default="mean")
        Test statistic to use for changepoint detection.
        * If "mean", the difference-in-mean statistic is used,
        * If "var", the difference-in-variance statistic is used,
        * If a tuple, it must contain two functions: The first function is the scoring
        function, which takes in the output of the second function as its first
        argument, and start, end and split indices as the second, third and fourth
        arguments. The second function is the initializer, which precomputes quantities
        that should be precomputed. See skchange/scores/score_factory.py for examples.
    threshold : float, optional (default=None)
        Threshold to use for changepoint detection.
        * If None, the threshold is set to the default value for the test statistic
        derived in [1]_.
        * If tune = True, the `threshold` input is ignored as it is tuned instead.
    min_interval_length : int (default=2)
        The minimum length of an interval to estimate a changepoint in.
    max_interval_length : int (default=200)
        The maximum length of an interval to estimate a changepoint in.
    growth_factor : float (default = 1.5)
        The growth factor for the seeded intervals. Intervals grow in size according to
        'interval_len=max(interval_len + 1, np.floor(growth_factor * interval_len))',
        starting at 'interval_len'='min_interval_length'. It also governs the amount
        of overlap between intervals of the same length, as the start of each interval
        is shifted by a factor of '1 + 1 / growth_factor'. Must be a float in (1, 2].
    tune : bool, optional (default=False)
        Whether to tune the threshold in the fit method or not. Takes precedence over
        the `threshold` input.

    References
    ----------
    .. [1] Kovács, S., Bühlmann, P., Li, H., & Munk, A. (2023). Seeded binary
    segmentation: a general methodology for fast and optimal changepoint detection.
    Biometrika, 110(1), 249-256.

    Examples
    --------
    from skchange.change_detectors.binary_segmentation import SeededBinarySegmentation
    from skchange.datasets.generate import teeth

    # Generate data
    df = teeth(n_segments=2, mean=10, segment_length=100000, p=5, random_state=2)

    # Detect changepoints
    detector = SeededBinarySegmentation()
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
        threshold: Optional[float] = None,
        min_interval_length: int = 2,
        max_interval_length: int = 200,
        growth_factor: float = 1.5,
        tune: bool = False,
        fmt: str = "sparse",
        labels: str = "int_label",
    ):
        self.score = score
        self.threshold = threshold  # Just holds the input value.
        self.min_interval_length = min_interval_length
        self.max_interval_length = max_interval_length
        self.growth_factor = growth_factor
        self.tune = tune
        super().__init__(fmt=fmt, labels=labels)
        self.score_f, self.score_init_f = score_factory(self.score)

        if threshold is not None and threshold < 0:
            raise ValueError(f"threshold must be non-negative (threshold={threshold}).")
        if self.min_interval_length < 2:
            raise ValueError(
                "min_interval_length must be at least 2"
                + f"(min_interval_length={self.min_interval_length})."
            )
        if self.max_interval_length < self.min_interval_length:
            raise ValueError(
                "max_interval_length must be at least min_interval_length"
                + f" (max_interval_length={self.max_interval_length}, "
            )
        if self.growth_factor <= 1 or self.growth_factor > 2:
            raise ValueError(
                "growth_factor must be in (1, 2]"
                + f" (growth_factor={self.growth_factor})."
            )

    def _check_X(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if X.isna().any(axis=None):
            raise ValueError("X must not contain missing values.")

        if X.ndim < 2:
            X = X.to_frame()

        if X.shape[0] < self.min_interval_length:
            raise ValueError(
                "X must have at least min_interval_length samples"
                + f" (X.shape[0]={X.shape[0]},"
                + f" min_interval_length={self.min_interval_length})."
            )
        return X

    def _tune_threshold(self, X: pd.DataFrame) -> float:
        """Tune the threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to tune the threshold on.

        Returns
        -------
        threshold : float
            The tuned threshold.
        """
        return None

    @staticmethod
    def get_default_threshold(n: int, p: int) -> float:
        """Get the default threshold for the Moscore algorithm.

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
        return 4 * p * np.sqrt(np.log(n))

    def _get_threshold(self, X: pd.DataFrame) -> float:
        # TODO:
        # if self.tune:
        #     return self._tune_threshold(X)

        if self.threshold:
            return self.threshold
        else:
            # The default threshold is used.
            return self.get_default_threshold(X.shape[0], X.shape[1])

    def _fit(self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None):
        """Fit to training data.

        Trains the threshold on the input data if `tune` is True. Otherwise, the
        threshold is set to the input `threshold` value if provided. If not,
        it is set to the default value for the test statistic, which depends on
        the dimension of X.

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
        """
        X = self._check_X(X)
        self.threshold_ = self._get_threshold(X)
        return self

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
        cpts, scores, maximizers, starts, ends = run_seeded_binary_segmentation(
            X.values,
            self.score_f,
            self.score_init_f,
            self.threshold_,
            self.min_interval_length,
            self.max_interval_length,
            self.growth_factor,
        )
        self.changepoints = cpts
        self.scores = pd.DataFrame(
            {"start": starts, "end": ends, "maximizer": maximizers, "score": scores}
        )
        self.per_sample_scores = (
            self.scores.groupby("maximizer")["score"]
            .max()
            .reindex(range(X.shape[0]), fill_value=0)
        ).values
        return format_changepoint_output(
            self.fmt, self.labels, self.changepoints, X.index, self.per_sample_scores
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
            {"score": "mean"},
            {"score": "mean", "threshold": 0},
        ]
        return params
