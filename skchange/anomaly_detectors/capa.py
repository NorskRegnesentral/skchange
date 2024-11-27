"""The collective and point anomalies (CAPA) algorithm."""

__author__ = ["Tveten"]
__all__ = ["CAPA"]

from typing import Optional, Union

import numpy as np
import pandas as pd

from skchange.anomaly_detectors.base import CollectiveAnomalyDetector
from skchange.anomaly_detectors.mvcapa import capa_penalty, run_base_capa
from skchange.anomaly_scores import BaseSaving, to_saving
from skchange.costs import BaseCost, L2Cost
from skchange.utils.validation.data import check_data
from skchange.utils.validation.parameters import check_larger_than


def run_capa(
    X: np.ndarray,
    collective_saving: BaseSaving,
    point_saving: BaseSaving,
    collective_alpha: float,
    point_alpha: float,
    min_segment_length: int,
    max_segment_length: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    collective_betas = np.zeros(1)
    point_betas = np.zeros(1)
    collective_saving.fit(X)
    point_saving.fit(X)
    return run_base_capa(
        collective_saving,
        point_saving,
        collective_alpha,
        collective_betas,
        point_alpha,
        point_betas,
        min_segment_length,
        max_segment_length,
    )


class CAPA(CollectiveAnomalyDetector):
    """Collective and point anomaly detection.

    An efficient implementation of the CAPA algorithm [1]_ for anomaly detection.
    It is implemented using the 'savings' formulation of the problem given in [2]_ and
    [3]_.

    `CAPA` can be applied to both univariate and multivariate data, but does not infer
    the subset of affected components for each anomaly in the multivariate case. See
    `MVCAPA` if such inference is desired.

    Parameters
    ----------
    collective_saving : BaseSaving or BaseCost, optional (default=L2Cost(0.0))
        The saving function to use for collective anomaly detection.
        If a `BaseCost` is given, the saving function is constructed from the cost. The
        cost must have a fixed parameter that represents the baseline cost.
    point_saving : BaseSaving or BaseCost, optional (default=L2Cost(0.0))
        The saving function to use for point anomaly detection. Only savings with a
        minimum size of 1 are permitted.
        If a `BaseCost` is given, the saving function is constructed from the cost. The
        cost must have a fixed parameter that represents the baseline cost.
    collective_penalty_scale : float, optional (default=2.0)
        Scaling factor for the collective penalty.
    point_penalty_scale : float, optional (default=2.0)
        Scaling factor for the point penalty.
    min_segment_length : int, optional (default=2)
        Minimum length of a segment.
    max_segment_length : int, optional (default=1000)
        Maximum length of a segment.
    ignore_point_anomalies : bool, optional (default=False)
        If True, detected point anomalies are not returned by `predict`. I.e., only
        collective anomalies are returned. If False, point anomalies are included in the
        output as collective anomalies of length 1.

    See Also
    --------
    MVCAPA : Multivariate CAPA with subset inference.

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
    0    [100, 199]
    1    [300, 399]
    Name: anomaly_interval, dtype: interval
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        collective_saving: Union[BaseSaving, BaseCost] = L2Cost(0.0),
        point_saving: Union[BaseSaving, BaseCost] = L2Cost(0.0),
        collective_penalty_scale: float = 2.0,
        point_penalty_scale: float = 2.0,
        min_segment_length: int = 2,
        max_segment_length: int = 1000,
        ignore_point_anomalies: bool = False,
    ):
        self.collective_saving = collective_saving
        self.point_saving = point_saving
        self.collective_penalty_scale = collective_penalty_scale
        self.point_penalty_scale = point_penalty_scale
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.ignore_point_anomalies = ignore_point_anomalies
        super().__init__()

        self._collective_saving = to_saving(collective_saving)

        if point_saving.min_size is not None and point_saving.min_size > 1:
            raise ValueError("Point saving must have a minimum size of 1.")
        self._point_saving = to_saving(point_saving)

        check_larger_than(0, collective_penalty_scale, "collective_penalty_scale")
        check_larger_than(0, point_penalty_scale, "point_penalty_scale")
        check_larger_than(2, min_segment_length, "min_segment_length")
        check_larger_than(min_segment_length, max_segment_length, "max_segment_length")

    def _get_penalty_components(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        # TODO: Add penalty tuning.
        # if self.tune:
        #     return self._tune_threshold(X)
        n = X.shape[0]
        p = X.shape[1]
        n_params = self._collective_saving.get_param_size(p)
        collective_penalty = capa_penalty(n, n_params, self.collective_penalty_scale)
        point_penalty = self.point_penalty_scale * n_params * p * np.log(n)
        return collective_penalty, point_penalty

    def _fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """Fit to training data.

        Sets the penalty of the detector.
        If `penalty_scale` is None, the penalty is set to the (1-`level`)-quantile
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
            Does nothing. Only here to make the fit method compatible with sktime
            and scikit-learn.

        Returns
        -------
        self : returns a reference to self

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """
        X = check_data(
            X,
            min_length=self.min_segment_length,
            min_length_name="min_segment_length",
        )
        self.collective_penalty_, self.point_penalty_ = self._get_penalty_components(X)
        return self

    def _predict(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Detect events in test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to detect events in (time series).

        Returns
        -------
        pd.Series[pd.Interval]
            Containing the collective anomaly intervals.

        Notes
        -----
        The start and end points of the intervals can be accessed by
        `output.array.left` and `output.array.right`, respectively.
        """
        X = check_data(
            X,
            min_length=self.min_segment_length,
            min_length_name="min_segment_length",
        )
        opt_savings, collective_anomalies, point_anomalies = run_capa(
            X.values,
            self._collective_saving,
            self._point_saving,
            self.collective_penalty_,
            self.point_penalty_,
            self.min_segment_length,
            self.max_segment_length,
        )
        self.scores = pd.Series(opt_savings, index=X.index, name="score")

        anomalies = collective_anomalies
        if not self.ignore_point_anomalies:
            anomalies += point_anomalies
        anomalies = sorted(anomalies)

        return CollectiveAnomalyDetector._format_sparse_output(anomalies)

    def _score_transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Compute the CAPA scores for the input data.

        Parameters
        ----------
        X : pd.DataFrame - data to compute scores for, time series

        Returns
        -------
        scores : pd.Series - scores for sequence X

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
                "collective_saving": L2Cost(param=0.0),
                "point_saving": L2Cost(param=0.0),
                "min_segment_length": 5,
                "max_segment_length": 100,
            },
            {
                "collective_saving": L2Cost(param=0.0),
                "point_saving": L2Cost(param=0.0),
                "min_segment_length": 2,
                "max_segment_length": 20,
            },
        ]
        return params
