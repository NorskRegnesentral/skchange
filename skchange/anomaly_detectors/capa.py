"""The collective and point anomalies (CAPA) algorithm."""

__author__ = ["Tveten"]
__all__ = ["CAPA"]

from typing import Optional, Union

import numpy as np
import pandas as pd

from skchange.anomaly_detectors.base import BaseSegmentAnomalyDetector
from skchange.anomaly_detectors.mvcapa import run_base_capa
from skchange.anomaly_scores import BaseSaving, L2Saving, to_saving
from skchange.costs import BaseCost
from skchange.penalties import BasePenalty, ChiSquarePenalty, as_constant_penalty
from skchange.utils.validation.data import check_data
from skchange.utils.validation.parameters import check_larger_than
from skchange.utils.validation.penalties import check_constant_penalty


def run_capa(
    X: np.ndarray,
    segment_saving: BaseSaving,
    point_saving: BaseSaving,
    segment_alpha: float,
    point_alpha: float,
    min_segment_length: int,
    max_segment_length: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    segment_betas = np.zeros(1)
    point_betas = np.zeros(1)
    segment_saving.fit(X)
    point_saving.fit(X)
    return run_base_capa(
        segment_saving,
        point_saving,
        segment_alpha,
        segment_betas,
        point_alpha,
        point_betas,
        min_segment_length,
        max_segment_length,
    )


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
    segment_saving : BaseSaving or BaseCost, optional, default=L2Saving()
        The saving function to use for segment anomaly detection.
        If a `BaseCost` is given, the saving function is constructed from the cost. The
        cost must have a fixed parameter that represents the baseline cost.
    point_saving : BaseSaving or BaseCost, optional, default=L2Saving()
        The saving function to use for point anomaly detection. Only savings with a
        minimum size of 1 are permitted.
        If a `BaseCost` is given, the saving function is constructed from the cost. The
        cost must have a fixed parameter that represents the baseline cost.
    segment_penalty : Union[BasePenalty, float], optional, default=`ChiSquarePenalty`
        The penalty to use for the changepoint detection. If a float is given, it is
        interpreted as a constant penalty. If `None`, the penalty is set to a BIC
        penalty with ``n=X.shape[0]`` and
        ``n_params=segment_saving.get_param_size(X.shape[1])``,
        where ``X`` is the input data to `fit`.
    point_penalty : Union[BasePenalty, float], optional, default=`ChiSquarePenalty`
        The penalty to use for the changepoint detection. If a float is given, it is
        interpreted as a constant penalty. If `None`, the penalty is set to a BIC
        penalty with ``n=X.shape[0]`` and
        ``n_params=point_saving.get_param_size(X.shape[1])``,
        where ``X`` is the input data to `fit`.
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
    >>> from skchange.datasets.generate import generate_alternating_data
    >>> df = generate_alternating_data(n_segments=5, mean=10, segment_length=100)
    >>> detector = CAPA()
    >>> detector.fit_predict(df)
    0    [100, 200)
    1    [300, 400)
    Name: anomaly_interval, dtype: interval
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        segment_saving: Optional[Union[BaseSaving, BaseCost]] = None,
        point_saving: Optional[Union[BaseSaving, BaseCost]] = None,
        segment_penalty: Union[BasePenalty, float, None] = None,
        point_penalty: Union[BasePenalty, float, None] = None,
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
        self._segment_saving = to_saving(_segment_saving)

        _point_saving = L2Saving() if point_saving is None else point_saving
        if _point_saving.min_size is not None and _point_saving.min_size > 1:
            raise ValueError("Point saving must have a minimum size of 1.")
        self._point_saving = to_saving(_point_saving)

        check_constant_penalty(self.segment_penalty, caller=self, allow_none=True)
        check_constant_penalty(self.point_penalty, caller=self, allow_none=True)
        check_larger_than(2, min_segment_length, "min_segment_length")
        check_larger_than(min_segment_length, max_segment_length, "max_segment_length")

    def _fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
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

        n = X.shape[0]
        p = X.shape[1]
        segment_n_params = self._segment_saving.get_param_size(p)
        self.segment_penalty_ = (
            ChiSquarePenalty(n, segment_n_params)
            if self.segment_penalty is None
            else as_constant_penalty(self.segment_penalty)
        )
        point_n_params = self._point_saving.get_param_size(p)
        self.point_penalty_ = (
            ChiSquarePenalty(n, point_n_params)
            if self.point_penalty is None
            else as_constant_penalty(self.point_penalty)
        )
        return self

    def _predict(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
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
        opt_savings, segment_anomalies, point_anomalies = run_capa(
            X.values,
            self._segment_saving,
            self._point_saving,
            self.segment_penalty_.values[0],
            self.point_penalty_.values[0],
            self.min_segment_length,
            self.max_segment_length,
        )
        self.scores = pd.Series(opt_savings, index=X.index, name="score")

        anomalies = segment_anomalies
        if not self.ignore_point_anomalies:
            anomalies += point_anomalies
        anomalies = sorted(anomalies)

        return self._format_sparse_output(anomalies)

    def _transform_scores(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
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
