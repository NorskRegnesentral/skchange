"""The pruned exact linear time (PELT) algorithm."""

__author__ = ["mtveten"]
__all__ = ["Pelt"]


from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit
from sktime.annotation.base import BaseSeriesAnnotator

from skchange.change_detectors.utils import format_changepoint_output
from skchange.costs.cost_factory import cost_factory
from skchange.utils.validation.data import check_data
from skchange.utils.validation.parameters import check_larger_than


@njit
def get_changepoints(prev_cpts: list) -> list:
    changepoints = []
    i = len(prev_cpts) - 1
    while i >= 0:
        cpt_i = prev_cpts[i]
        changepoints.append(i)
        i = cpt_i
    return changepoints[:0:-1]  # Remove the artificial changepoint at the end


@njit
def run_pelt(
    X: np.ndarray, cost_func, cost_init_func, penalty, min_segment_length
) -> Tuple[np.ndarray, list]:
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

    return opt_cost[1:], get_changepoints(prev_cpts)


class Pelt(BaseSeriesAnnotator):
    """Pruned exact linear time changepoint detection.

    An efficient implementation of the PELT algorithm [1]_ for changepoint detection.

    Parameters
    ----------
    cost : str or callable, optional (default="mean")
        Cost function to use for changepoint detection.
        * If "mean", the Gaussian mean likelihood cost is used,
        * ...
    penalty_scale : float, optional (default=2.0)
        Scaling factor for the penalty. The penalty is set to
        'penalty_scale * 2 * p * np.log(n)', where 'n' is the sample size
        and 'p' is the number of variables. If None, the threshold is tuned on the data
        input to .fit() (not supported yet).
    min_segment_length : int, optional (default=2)
        Minimum length of a segment.
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
        "fit_is_empty": False,
    }

    def __init__(
        self,
        cost: Union[str, Callable] = "mean",
        penalty_scale: Optional[float] = 2.0,
        min_segment_length: int = 2,
        fmt: str = "sparse",
        labels: str = "int_label",
    ):
        self.cost = cost
        self.penalty_scale = penalty_scale
        self.min_segment_length = min_segment_length
        super().__init__(fmt=fmt, labels=labels)
        self.cost_func, self.cost_init_func = cost_factory(self.cost)

        check_larger_than(0, penalty_scale, "penalty_scale", allow_none=True)
        check_larger_than(1, min_segment_length, "min_segment_length")

    def _tune_penalty(self, X: pd.DataFrame) -> float:
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
        raise ValueError(
            "tuning of the penalty is not supported yet (penalty_scale=None)."
        )

    @staticmethod
    def get_default_penalty(n: int, p: int) -> float:
        """Get the default, BIC-penalty for PELT.

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
        return 2 * p * np.log(n)

    def _get_penalty(self, X: pd.DataFrame) -> float:
        if self.penalty_scale is None:
            return self._tune_penalty(X)
        else:
            n = X.shape[0]
            p = X.shape[1]
            return self.penalty_scale * self.get_default_penalty(n, p)

    def _fit(self, X: Union[pd.Series, pd.DataFrame], Y: Optional[pd.DataFrame] = None):
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
        X = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="2*min_segment_length",
        )
        self.threshold_ = self._get_penalty(X)
        return self

    def _predict(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame - data to annotate, time series

        Returns
        -------
        Y : pd.Series - annotations for sequence X
            exact format depends on annotation type
        """
        X = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="2*min_segment_length",
        )
        self._penalty = self._get_penalty(X)  # In case no penalty yet, use default.
        self.scores, self.changepoints = run_pelt(
            X.values,
            self.cost_func,
            self.cost_init_func,
            self._penalty,
            self.min_segment_length,
        )
        return format_changepoint_output(
            self.fmt, self.labels, self.changepoints, X.index, self.scores
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
            {"cost": "mean", "min_segment_length": 2},
            {"cost": "mean", "penalty_scale": 0, "min_segment_length": 1},
        ]
        return params
