"""The subset multivariate collective and point anomalies (MVCAPA) algorithm."""

__author__ = ["Tveten"]
__all__ = ["MVCAPA"]

import numpy as np
import pandas as pd

from skchange.anomaly_detectors.base import BaseSegmentAnomalyDetector
from skchange.anomaly_detectors.capa import run_capa
from skchange.anomaly_scores import BaseSaving, L2Saving, to_saving
from skchange.compose import PenalisedScore
from skchange.costs import BaseCost
from skchange.penalties import (
    BasePenalty,
    ChiSquarePenalty,
    LinearChiSquarePenalty,
    MinimumPenalty,
    NonlinearChiSquarePenalty,
    as_penalty,
)
from skchange.utils.validation.data import check_data
from skchange.utils.validation.parameters import check_larger_than


def find_affected_components(
    saving: BaseSaving, anomalies: list[tuple[int, int]], penalty: BasePenalty
) -> list[tuple[int, int, np.ndarray]]:
    saving.check_is_fitted()
    new_anomalies = []
    for start, end in anomalies:
        saving_values = saving.evaluate(np.array([start, end]))[0]
        saving_order = np.argsort(-saving_values)  # Decreasing order.
        penalised_saving = np.cumsum(saving_values[saving_order]) - penalty.values
        argmax = np.argmax(penalised_saving)
        new_anomalies.append((start, end, saving_order[: argmax + 1]))
    return new_anomalies


def run_mvcapa(
    X: np.ndarray,
    segment_penalised_saving: PenalisedScore,
    point_penalised_saving: PenalisedScore,
    min_segment_length: int,
    max_segment_length: int,
) -> tuple[
    np.ndarray, list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]
]:
    opt_savings, segment_anomalies, point_anomalies = run_capa(
        X,
        segment_penalised_saving=segment_penalised_saving,
        point_penalised_saving=point_penalised_saving,
        min_segment_length=min_segment_length,
        max_segment_length=max_segment_length,
    )
    segment_anomalies = find_affected_components(
        saving=segment_penalised_saving.scorer,
        anomalies=segment_anomalies,
        penalty=segment_penalised_saving.penalty,
    )
    point_anomalies = find_affected_components(
        saving=point_penalised_saving.scorer,
        anomalies=point_anomalies,
        penalty=point_penalised_saving.penalty,
    )
    return opt_savings, segment_anomalies, point_anomalies


class MVCAPA(BaseSegmentAnomalyDetector):
    """Subset multivariate collective and point anomaly detection.

    The MVCAPA algorithm [1]_ for anomaly detection.

    Parameters
    ----------
    segment_saving : BaseSaving or BaseCost, optional, default=L2Saving()
        The saving function to use for segment anomaly detection.
        Only univariate savings are permitted (see the `evaluation_type` attribute).
        If a `BaseCost` is given, the saving function is constructed from the cost. The
        cost must have a fixed parameter that represents the baseline cost.
    point_saving : BaseSaving or BaseCost, optional, default=L2Saving()
        The saving function to use for point anomaly detection. Only savings with a
        minimum size of 1 are permitted.
        If a `BaseCost` is given, the saving function is constructed from the cost. The
        cost must have a fixed parameter that represents the baseline cost.
    segment_penalty : BasePenalty, np.ndarray or float, optional, default=`MinimumPenalty([ChiSquarePenalty(), LinearChiSquarePenalty(), NonlinearChiSquarePenalty()])`
        The penalty to use for segment anomaly detection. If a float is given, it is
        interpreted as a constant penalty. If `None`, the default penalty is fit to the
        input data to `fit`.
    point_penalty : BasePenalty, np.ndarray or float, optional, default=`LinearChiSquarePenalty`
        The penalty to use for point anomaly detection. If a float is given, it is
        interpreted as a constant penalty. If `None`, the default penalty is fit to the
        input data to `fit`.
    min_segment_length : int, optional, default=2
        Minimum length of a segment.
    max_segment_length : int, optional, default=1000
        Maximum length of a segment.
    ignore_point_anomalies : bool, optional, default=False
        If True, detected point anomalies are not returned by `predict`. I.e., only
        segment anomalies are returned.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       segment and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.anomaly_detectors import MVCAPA
    >>> from skchange.datasets.generate import generate_anomalous_data
    >>> n = 300
    >>> means = [np.array([8.0, 0.0, 0.0]), np.array([2.0, 3.0, 5.0])]
    >>> df = generate_anomalous_data(
    >>>     n, anomalies=[(100, 120), (250, 300)], means=means, random_state=3
    >>> )
    >>> detector = MVCAPA()
    >>> detector.fit_predict(df)
      anomaly_interval anomaly_columns
    0       [100, 120)             [0]
    1       [250, 300)       [2, 1, 0]
    """  # noqa: E501

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }
    capability_variable_identification = True

    def __init__(
        self,
        segment_saving: BaseSaving | BaseCost | None = None,
        point_saving: BaseSaving | BaseCost | None = None,
        segment_penalty: BasePenalty | np.ndarray | float | None = None,
        point_penalty: BasePenalty | np.ndarray | float | None = None,
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
        if _segment_saving.evaluation_type == "multivariate":
            raise ValueError("Segment saving must be univariate.")
        self._segment_saving = to_saving(_segment_saving)

        _point_saving = L2Saving() if point_saving is None else point_saving
        if _point_saving.min_size != 1:
            raise ValueError("Point saving must have a minimum size of 1.")
        self._point_saving = to_saving(_point_saving)

        default_segment_penalty = MinimumPenalty(
            [ChiSquarePenalty(), LinearChiSquarePenalty(), NonlinearChiSquarePenalty()]
        )
        self._segment_penalty = as_penalty(
            self.segment_penalty,
            default=default_segment_penalty,
        )
        self._point_penalty = as_penalty(
            self.point_penalty,
            default=LinearChiSquarePenalty(),
        )

        check_larger_than(2, min_segment_length, "min_segment_length")
        check_larger_than(min_segment_length, max_segment_length, "max_segment_length")

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None):
        """Fit to training data.

        Sets the penalty of the detector.
        If `penalty_scale` is ``None``, the penalty is set to the ``1-level`` quantile
        of the change/anomaly scores on the training data. For this to be correct,
        the training data must contain no changepoints. If `penalty_scale` is a
        number, the penalty is set to `penalty_scale` times the default penalty
        for the detector. The default penalty depends at least on the data's shape,
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
        self :
            Reference to self.

        State change
        ------------
        Creates fitted model that updates attributes ending in "_".
        """
        X = check_data(
            X,
            min_length=self.min_segment_length,
            min_length_name="min_segment_length",
        )
        self.segment_penalty_ = self._segment_penalty.fit(X, self._segment_saving)
        self.point_penalty_ = self._point_penalty.fit(X, self._point_saving)

        self.segment_penalised_saving_ = PenalisedScore(
            self._segment_saving, self.segment_penalty_
        )
        self.segment_penalised_saving_.fit(X)

        self.point_penalised_saving_ = PenalisedScore(
            self._point_saving, self.point_penalty_
        )
        self.point_penalised_saving_.fit(X)

        return self

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
        opt_savings, segment_anomalies, point_anomalies = run_mvcapa(
            X.values,
            segment_penalised_saving=self.segment_penalised_saving_,
            point_penalised_saving=self.point_penalised_saving_,
            min_segment_length=self.min_segment_length,
            max_segment_length=self.max_segment_length,
        )
        self.scores = pd.Series(opt_savings, index=X.index, name="score")

        anomalies = segment_anomalies
        if not self.ignore_point_anomalies:
            anomalies += point_anomalies
        anomalies = sorted(anomalies)

        return self._format_sparse_output(anomalies)

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

        Notes
        -----
        The `MVCAPA` scores are the cumulative optimal savings, so the scores are
        increasing and are not per observation scores.
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
