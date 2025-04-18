"""The collective and point anomalies (CAPA) algorithm."""

__author__ = ["Tveten"]
__all__ = ["CAPA"]

import numpy as np
import pandas as pd

from ..anomaly_scores import L2Saving, to_saving
from ..base import BaseIntervalScorer
from ..compose.penalised_score import PenalisedScore
from ..penalties import ChiSquarePenalty, as_penalty
from ..penalties.base import BasePenalty
from ..utils.numba import njit
from ..utils.validation.data import check_data
from ..utils.validation.parameters import check_larger_than
from .base import BaseSegmentAnomalyDetector


@njit
def get_anomalies(
    anomaly_starts: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    segment_anomalies = []
    point_anomalies = []
    i = anomaly_starts.size - 1
    while i >= 0:
        start_i = anomaly_starts[i]
        size = i - start_i + 1
        if size > 1:
            segment_anomalies.append((int(start_i), i + 1))
            i = int(start_i)
        elif size == 1:
            point_anomalies.append((i, i + 1))
        i -= 1
    return segment_anomalies, point_anomalies


def run_capa(
    segment_penalised_saving: PenalisedScore,
    point_penalised_saving: PenalisedScore,
    min_segment_length: int,
    max_segment_length: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    segment_penalised_saving.check_is_penalised()
    segment_penalised_saving.check_is_fitted()
    point_penalised_saving.check_is_penalised()
    point_penalised_saving.check_is_fitted()
    n_samples = segment_penalised_saving._X.shape[0]
    if not n_samples == point_penalised_saving._X.shape[0]:
        raise ValueError(
            "The segment and point saving costs must span the same number of samples."
        )

    opt_savings = np.zeros(n_samples + 1)

    # Store the optimal start and affected components of an anomaly for each t.
    # Used to get the final set of anomalies after the loop.
    opt_anomaly_starts = np.repeat(np.nan, n_samples)
    starts = np.array([], dtype=int)

    ts = np.arange(min_segment_length - 1, n_samples)
    for t in ts:
        # Segment anomalies
        t_array = np.array([t])

        starts = np.concatenate((starts, t_array - min_segment_length + 1))
        ends = np.repeat(t + 1, len(starts))
        intervals = np.column_stack((starts, ends))
        segment_savings = segment_penalised_saving.evaluate(intervals).flatten()
        candidate_savings = opt_savings[starts] + segment_savings
        candidate_argmax = np.argmax(candidate_savings)
        opt_segment_saving = candidate_savings[candidate_argmax]
        opt_start = starts[0] + candidate_argmax

        # Point anomalies
        point_interval = np.column_stack((t_array, t_array + 1))
        point_savings = point_penalised_saving.evaluate(point_interval).flatten()
        opt_point_saving = opt_savings[t] + point_savings[0]

        # Combine and store results
        savings = np.array([opt_savings[t], opt_segment_saving, opt_point_saving])
        argmax = np.argmax(savings)
        opt_savings[t + 1] = savings[argmax]
        if argmax == 1:
            opt_anomaly_starts[t] = opt_start
        elif argmax == 2:
            opt_anomaly_starts[t] = t

        # Pruning the admissible starts
        penalty_sum = segment_penalised_saving.penalty_.values[-1]
        saving_too_low = candidate_savings + penalty_sum <= opt_savings[t + 1]
        too_long_segment = starts < t - max_segment_length + 2
        prune = saving_too_low | too_long_segment
        starts = starts[~prune]

    segment_anomalies, point_anomalies = get_anomalies(opt_anomaly_starts)
    return opt_savings[1:], segment_anomalies, point_anomalies


class CAPA(BaseSegmentAnomalyDetector):
    """The collective and point anomaly (CAPA) detection algorithm.

    An efficient implementation of the CAPA algorithm [1]_ for anomaly detection.
    It is implemented using the 'savings' formulation of the problem given in [2]_ and
    [3]_.

    `CAPA` can be applied to both univariate and multivariate data, but does not infer
    the subset of affected components for each anomaly in the multivariate case. See
    `MVCAPA` if such inference is desired.

    Parameters
    ----------
    segment_saving : BaseIntervalScorer, optional, default=L2Saving()
        The saving to use for segment anomaly detection.
        If a cost is given, the saving is constructed from the cost. The
        cost must have a fixed parameter that represents the baseline cost.
    point_saving : BaseIntervalScorer, optional, default=L2Saving()
        The saving to use for point anomaly detection. Only savings with a
        minimum size of 1 are permitted.
        If a cost is given, the saving is constructed from the cost. The
        cost must have a fixed parameter that represents the baseline cost.
    segment_penalty : BasePenalty or float, optional, default=`ChiSquarePenalty`
        The penalty to use for segment anomaly detection. If a float is given, it is
        interpreted as a constant penalty. If `None`, the default penalty is fit to the
        input data to `predict`.
    point_penalty : BasePenalty or float, optional, default=`ChiSquarePenalty`
        The penalty to use for point anomaly detection. If a float is given, it is
        interpreted as a constant penalty. If `None`, the default penalty is fit to the
        input data to `predict`.
    min_segment_length : int, optional, default=2
        Minimum length of a segment.
    max_segment_length : int, optional, default=1000
        Maximum length of a segment.
    ignore_point_anomalies : bool, optional, default=False
        If ``True``, detected point anomalies are not returned by `predict`. I.e., only
        segment anomalies are returned. If ``False``, point anomalies are included in
        the output as segment anomalies of length 1.

    See Also
    --------
    MVCAPA : Multivariate CAPA with affected variable inference.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). A linear time method\
        for the detection of collective and point anomalies. Statistical Analysis and\
        DataMining: The ASA Data Science Journal, 15(4), 494-508.

    .. [2] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate\
        collective and point anomaly detection. Journal of Computational and Graphical\
        Statistics, 31(2), 574-585.

    .. [3] Tveten, M., Eckley, I. A., & Fearnhead, P. (2022). Scalable change-point and\
        anomaly detection in cross-correlated data with an application to condition\
        monitoring. The Annals of Applied Statistics, 16(2), 721-743.

    Examples
    --------
    >>> from skchange.anomaly_detectors import CAPA
    >>> from skchange.datasets import generate_alternating_data
    >>> df = generate_alternating_data(n_segments=5, mean=10, segment_length=100)
    >>> detector = CAPA()
    >>> detector.fit_predict(df)
            ilocs  labels
    0  [100, 200)       1
    1  [300, 400)       2
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        segment_saving: BaseIntervalScorer | None = None,
        point_saving: BaseIntervalScorer | None = None,
        segment_penalty: BasePenalty | float | None = None,
        point_penalty: BasePenalty | float | None = None,
        min_segment_length: int = 2,
        max_segment_length: int = 1000,
        ignore_point_anomalies: bool = False,
    ):
        self.segment_saving = segment_saving
        self.point_saving = point_saving
        self.segment_penalty = segment_penalty
        self.point_penalty = point_penalty
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.ignore_point_anomalies = ignore_point_anomalies
        super().__init__()

        _segment_saving = L2Saving() if segment_saving is None else segment_saving
        _segment_saving = to_saving(_segment_saving)
        _segment_penalty = as_penalty(
            self.segment_penalty,
            default=ChiSquarePenalty(),
            require_penalty_type="constant",
        )
        self._segment_penalised_saving = PenalisedScore(
            _segment_saving, _segment_penalty
        )

        _point_saving = L2Saving() if point_saving is None else point_saving
        if _point_saving.min_size != 1:
            raise ValueError("Point saving must have a minimum size of 1.")
        _point_saving = to_saving(_point_saving)
        _point_penalty = as_penalty(
            self.point_penalty,
            default=ChiSquarePenalty(),
            require_penalty_type="constant",
        )
        self._point_penalised_saving = PenalisedScore(_point_saving, _point_penalty)

        check_larger_than(2, min_segment_length, "min_segment_length")
        check_larger_than(min_segment_length, max_segment_length, "max_segment_length")

        self.set_tags(distribution_type=_segment_saving.get_tag("distribution_type"))

    def _predict(self, X: pd.DataFrame | pd.Series) -> pd.Series:
        """Detect events in test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series to detect anomalies in.

        Returns
        -------
        y_sparse: pd.DataFrame
            A `pd.DataFrame` with a range index and two columns:
            * ``"ilocs"`` - left-closed ``pd.Interval``s of iloc based segments.
            * ``"labels"`` - integer labels ``1, ..., K`` for each segment anomaly.
        """
        X = check_data(
            X,
            min_length=self.min_segment_length,
            min_length_name="min_segment_length",
        )

        self._segment_penalised_saving.fit(X)
        self._point_penalised_saving.fit(X)
        opt_savings, segment_anomalies, point_anomalies = run_capa(
            segment_penalised_saving=self._segment_penalised_saving,
            point_penalised_saving=self._point_penalised_saving,
            min_segment_length=self.min_segment_length,
            max_segment_length=self.max_segment_length,
        )
        self.scores = pd.Series(opt_savings, index=X.index, name="score")

        anomalies = segment_anomalies
        if not self.ignore_point_anomalies:
            anomalies += point_anomalies
        anomalies = sorted(anomalies)

        return self._format_sparse_output(anomalies)

    def _transform_scores(self, X: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Return scores for predicted labels on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to score (time series).

        Returns
        -------
        scores : pd.DataFrame with same index as X
            Scores for sequence `X`.

        Notes
        -----
        The CAPA scores are the cumulative optimal savings, so the scores are increasing
        and are not per observation scores.
        """
        self.predict(X)
        return self.scores

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
            {
                "segment_saving": L2Cost(param=0.0),
                "point_saving": L2Cost(param=0.0),
                "min_segment_length": 5,
                "max_segment_length": 100,
            },
            {
                "segment_saving": L2Cost(param=0.0),
                "point_saving": L2Cost(param=0.0),
                "min_segment_length": 2,
                "max_segment_length": 20,
            },
        ]
        return params
