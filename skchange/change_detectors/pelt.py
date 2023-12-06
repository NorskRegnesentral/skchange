"""The pruned exact linear time (PELT) algorithm."""

__author__ = ["mtveten"]
__all__ = ["Pelt"]


from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from numba import njit
from sktime.annotation.base import BaseSeriesAnnotator

from skchange.change_detectors.utils import changepoints_to_labels
from skchange.costs.cost_factory import cost_factory


def BIC_penalty(n: int, n_params: int):
    return n_params * np.log(n)


@njit
def get_changepoints(prev_cpts: list) -> list:
    changepoints = []
    i = len(prev_cpts) - 1
    while i >= 0:
        cpt_i = prev_cpts[i]
        changepoints.append(i)
        i = cpt_i
    return changepoints[::-1]


@njit
def run_pelt(X: np.ndarray, cost_func, cost_init_func, penalty, min_segment_length):
    params = cost_init_func(X)
    n = len(X)

    admissible = np.array([0])
    opt_cost = np.zeros(n + 1)
    opt_cost[: min_segment_length - 1] = -penalty

    # Store the previous changepoint for each t.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = [-1] * (min_segment_length - 1)

    ts = np.arange(min_segment_length - 1, n).reshape(-1, 1)
    for t in ts:
        new_admissible = t - min_segment_length + 1
        admissible = np.concatenate((admissible, new_admissible))
        ends = np.repeat(t, len(admissible))
        admissible_opt_costs = (
            opt_cost[admissible] + cost_func(params, admissible, ends) + penalty
        )
        admissible_argmin = np.argmin(admissible_opt_costs)
        opt_cost[t] = admissible_opt_costs[admissible_argmin]
        prev_cpts.append(admissible[admissible_argmin] - 1)

        # trimming the admissible set
        admissible = admissible[admissible_opt_costs - penalty <= opt_cost[t]]

    return get_changepoints(prev_cpts)


class Pelt(BaseSeriesAnnotator):
    """Pruned exact linear time changepoint detection.

    An efficient implementation of the PELT algorithm [1]_ for changepoint detection.

    Parameters
    ----------
    fmt : str {"dense", "sparse"}, optional (default="sparse")
        Annotation output format:
        * If "sparse", a sub-series of labels for only the outliers in X is returned,
        * If "dense", a series of labels for all values in X is returned.
    labels : str {"indicator", "score"}, optional (default="indicator")
        Annotation output labels:
        * If "indicator", returned values are boolean, indicating whether a value is an
        outlier,
        * If "score", returned values are floats, giving the outlier score.
    cost : str or callable, optional (default="l2")
        Cost function to use for changepoint detection.
        * If "l2", the l2-norm is used,
        * ...
    penalty : float, optional (default=None)
        Penalty to use for changepoint detection.
        * If None, the penalty is set to log(n) * p, where n is the number of samples
        and p is the number of dimensions in X.
    min_segment_length : int, optional (default=2)
        Minimum length of a segment.


    References
    ----------
    .. [1] Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of
    changepoints with a linear computational cost. Journal of the American Statistical
    Association, 107(500), 1590-1598.

    Examples
    --------
    from skchange.change_detectors.pelt import Pelt
    from skchange.datasets.generate import teeth

    # Generate data
    df = teeth(n_segments=2, mean=10, segment_length=100000, p=5, random_state=2)

    # skchange method
    detector = Pelt()
    detector.fit_predict(df)
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        cost: Union[str, Callable] = "l2",
        penalty: Optional[float] = None,
        min_segment_length: int = 2,
        fmt: str = "sparse",
        labels: str = "indicator",
    ):
        self.cost = cost
        self.penalty = penalty
        self.min_segment_length = min_segment_length

        super().__init__(fmt=fmt, labels=labels)

        self.cost_func, self.cost_init_func = cost_factory(self.cost)

        if self.min_segment_length < 1:
            raise ValueError("min_segment_length must be at least 1.")

    def _fit(self, X, Y=None):
        """Fit to training data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            training data to fit model to, time series
        Y : pd.Series, optional
            ground truth annotations for training if annotator is supervised
        Returns
        -------
        self : returns a reference to self

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """
        return self

    def _get_penalty(self, X: pd.DataFrame) -> float:
        n = X.shape[0]
        p = X.shape[1]
        penalty = self.penalty if self.penalty else BIC_penalty(n, p)
        if penalty < 0:
            raise ValueError(f"penalty must be non-negative (penalty={self.penalty}).")
        return penalty

    def _format_predict_output(self, changepoints, X_index):
        if self.fmt == "sparse":
            return np.array(changepoints)
        else:
            labels = changepoints_to_labels(changepoints)
            return pd.Series(labels, index=X_index)

    def _predict(self, X):
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
        if X.ndim < 2:
            X = X.to_frame()

        self._penalty = self._get_penalty(X)  # In case no penalty yet, use default.
        changepoints = run_pelt(
            X.values,
            self.cost_func,
            self.cost_init_func,
            self._penalty,
            self.min_segment_length,
        )
        return self._format_predict_output(changepoints, X.index)

    # todo: consider implementing this, optional
    # if not implementing, delete the _update method
    # def _update(self, X, Y=None):
    #     """Update model with new data and optional ground truth annotations.

    #     core logic

    #     Parameters
    #     ----------
    #     X : pd.DataFrame
    #         training data to update model with, time series
    #     Y : pd.Series, optional
    #         ground truth annotations for training if annotator is supervised
    #     Returns
    #     -------
    #     self : returns a reference to self

    #     State change
    #     ------------
    #     updates fitted model (attributes ending in "_")
    #     """

    # implement here
    # IMPORTANT: avoid side effects to X, fh

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
            {"cost": "l2", "penalty": None, "min_segment_length": 2},
            {"cost": "l2", "penalty": 0, "min_segment_length": 1},
        ]
        return params
