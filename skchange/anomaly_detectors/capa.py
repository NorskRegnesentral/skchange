"""The collective and point anomalies (CAPA) algorithm."""

__author__ = ["Tveten"]
__all__ = ["Capa"]

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from numba import njit

from skchange.anomaly_detectors.base import CollectiveAnomalyDetector
from skchange.anomaly_detectors.mvcapa import dense_capa_penalty, run_base_capa
from skchange.costs.saving_factory import saving_factory
from skchange.utils.validation.data import check_data
from skchange.utils.validation.parameters import check_larger_than


@njit
def run_capa(
    X: np.ndarray,
    saving_func: Callable,
    saving_init_func: Callable,
    collective_alpha: float,
    point_alpha: float,
    min_segment_length: int,
    max_segment_length: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    params = saving_init_func(X)
    collective_betas = np.zeros(1)
    point_betas = np.zeros(1)
    return run_base_capa(
        X,
        params,
        saving_func,
        collective_alpha,
        collective_betas,
        point_alpha,
        point_betas,
        min_segment_length,
        max_segment_length,
    )


class Capa(CollectiveAnomalyDetector):
    """Collective and point anomaly detection.

    An efficient implementation of the CAPA algorithm [1]_ for anomaly detection.
    It is implemented using the 'savings' formulation of the problem given in [2]_.

    Capa can be applied to both univariate and multivariate data, but does not infer
    the subset of affected components for each anomaly in the multivariate case. See the
    Mvcapa class if such inference is desired.

    Parameters
    ----------
    saving : str (default="mean")
        Saving function to use for anomaly detection.
    collective_penalty_scale : float, optional (default=2.0)
        Scaling factor for the collective penalty.
    point_penalty_scale : float, optional (default=2.0)
        Scaling factor for the point penalty.
    min_segment_length : int, optional (default=2)
        Minimum length of a segment.
    max_segment_length : int, optional (default=1000)
        Maximum length of a segment.
    ignore_point_anomalies : bool, optional (default=False)
        If True, detected point anomalies are not returned by .predict(). I.e., only
        collective anomalies are returned. If False, point anomalies are included in the
        output as collective anomalies of length 1.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). A linear time method\
    for the detection of collective and point anomalies. Statistical Analysis and Data\
    Mining: The ASA Data Science Journal, 15(4), 494-508.

    .. [2] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate\
    collective and point anomaly detection. Journal of Computational and Graphical\
    Statistics, 31(2), 574-585.

    Examples
    --------
    >>> from skchange.anomaly_detectors.capa import Capa
    >>> from skchange.datasets.generate import generate_teeth_data
    >>> df = generate_teeth_data(n_segments=5, mean=10, segment_length=100)
    >>> detector = Capa()
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
        saving: Union[str, tuple[Callable, Callable]] = "mean",
        collective_penalty_scale: float = 2.0,
        point_penalty_scale: float = 2.0,
        min_segment_length: int = 2,
        max_segment_length: int = 1000,
        ignore_point_anomalies: bool = False,
    ):
        self.saving = saving
        self.collective_penalty_scale = collective_penalty_scale
        self.point_penalty_scale = point_penalty_scale
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.ignore_point_anomalies = ignore_point_anomalies
        super().__init__()

        self.saving_func, self.saving_init_func = saving_factory(self.saving)

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
        # TODO: Add support for depending on 'score'. May interact with p.
        #       E.g. if score is multivariate normal with unknown covariance.
        n_params = 1
        # The default penalty is inflated by a factor of 2 as it is based on Gaussian
        # data. Most data is more heavy-tailed, so we use a bigger penalty.
        # In addition, false positive control is often more important than higher power.
        collective_penalty = dense_capa_penalty(
            n, p, n_params, self.collective_penalty_scale
        )[0]
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

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Data to detect events in (time series).

        Returns
        -------
        pd.Series[pd.Interval] containing the collective anomaly intervals.

        Notes
        -----
        The start and end points of the intervals can be accessed by
        output.array.left and output.array.right, respectively.
        """
        X = check_data(
            X,
            min_length=self.min_segment_length,
            min_length_name="min_segment_length",
        )
        opt_savings, collective_anomalies, point_anomalies = run_capa(
            X.values,
            self.saving_func,
            self.saving_init_func,
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
        params = [
            {"saving": "mean", "min_segment_length": 5, "max_segment_length": 100},
            {"saving": "mean", "min_segment_length": 2, "max_segment_length": 20},
        ]
        return params
