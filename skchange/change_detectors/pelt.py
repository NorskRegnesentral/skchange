"""The pruned exact linear time (PELT) algorithm."""

__author__ = ["Tveten"]
__all__ = ["Pelt"]


from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from skchange.change_detectors.base import ChangeDetector
from skchange.costs.cost_factory import cost_factory
from skchange.utils.numba import njit
from skchange.utils.validation.data import check_data
from skchange.utils.validation.parameters import check_larger_than


@njit
def get_changepoints(prev_cpts: np.ndarray) -> list:
    changepoints = []
    i = len(prev_cpts) - 1
    while i >= 0:
        cpt_i = prev_cpts[i]
        changepoints.append(i)
        i = cpt_i
    return changepoints[:0:-1]  # Remove the artifical changepoint at the last index.


@njit
def run_pelt(
    X: np.ndarray, cost_func, cost_init_func, penalty, min_segment_length
) -> tuple[np.ndarray, list]:
    params = cost_init_func(X)
    n = len(X)

    starts = np.array((), dtype=np.int64)  # Evolving set of admissible segment starts.
    init_starts = np.zeros(min_segment_length - 1, dtype=np.int64)
    init_ends = np.arange(min_segment_length - 1)
    opt_cost = np.zeros(n + 1) - penalty
    opt_cost[1:min_segment_length] = cost_func(params, init_starts, init_ends)

    # Store the previous changepoint for each t.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = np.repeat(-1, n)

    ts = np.arange(min_segment_length - 1, n).reshape(-1, 1)
    for t in ts:
        starts = np.concatenate((starts, t - min_segment_length + 1))
        ends = np.repeat(t, len(starts))
        candidate_opt_costs = (
            opt_cost[starts] + cost_func(params, starts, ends) + penalty
        )
        argmin = np.argmin(candidate_opt_costs)
        opt_cost[t + 1] = candidate_opt_costs[argmin]
        prev_cpts[t] = starts[argmin] - 1

        # Trimming the admissible starts set
        starts = starts[candidate_opt_costs - penalty <= opt_cost[t]]

    return opt_cost[1:], get_changepoints(prev_cpts)


class Pelt(ChangeDetector):
    """Pruned exact linear time changepoint detection.

    An efficient implementation of the PELT algorithm [1]_ for changepoint detection.

    Parameters
    ----------
    cost : {"mean"}, tuple[Callable, Callable], default="mean
        Name of cost function to use for changepoint detection.

        * `"mean"`: The Gaussian mean likelihood cost is used,
        * More cost functions will be added in the future.
    penalty_scale : float, optional (default=2.0)
        Scaling factor for the penalty. The penalty is set to
        `penalty_scale * 2 * p * np.log(n)`, where `n` is the sample size
        and `p` is the number of variables. If None, the penalty is tuned on the data
        input to `fit` (not supported yet).
    min_segment_length : int, optional (default=2)
        Minimum length of a segment.

    References
    ----------
    .. [1] Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of
    changepoints with a linear computational cost. Journal of the American Statistical
    Association, 107(500), 1590-1598.

    Examples
    --------
    >>> from skchange.change_detectors import Pelt
    >>> from skchange.datasets.generate import generate_alternating_data
    >>> df = generate_alternating_data(n_segments=2, mean=10, segment_length=10000, p=5)
    >>> detector = Pelt()
    >>> detector.fit_predict(df)
    0    9999
    Name: changepoint, dtype: int64
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
    ):
        self.cost = cost
        self.penalty_scale = penalty_scale
        self.min_segment_length = min_segment_length
        super().__init__()
        self.cost_func, self.cost_init_func = cost_factory(self.cost)

        check_larger_than(0, penalty_scale, "penalty_scale", allow_none=True)
        check_larger_than(1, min_segment_length, "min_segment_length")

    def _tune_penalty(self, X: pd.DataFrame) -> float:
        """Tune the penalty.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to tune the penalty on.

        Returns
        -------
        penalty : float
            The tuned penalty.
        """
        raise ValueError(
            "tuning of the penalty is not supported yet (`penalty_scale=None`)."
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
        penalty : float
            The default penalty.
        """
        return 2 * p * np.log(n)

    def _get_penalty(self, X: pd.DataFrame) -> float:
        if self.penalty_scale is None:
            return self._tune_penalty(X)
        else:
            n = X.shape[0]
            p = X.shape[1]
            return self.penalty_scale * self.get_default_penalty(n, p)

    def _fit(
        self,
        X: Union[pd.Series, pd.DataFrame],
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ):
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
            training data to fit the penalty to.
        y : pd.Series, optional
            Does nothing. Only here to make the `fit` method compatible with `sktime`
            and `scikit-learn`.

        Returns
        -------
        self : returns a reference to self
        """
        X = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="2*min_segment_length",
        )
        self.penalty_ = self._get_penalty(X)
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
            min_length_name="2*min_segment_length",
        )
        opt_costs, changepoints = run_pelt(
            X.values,
            self.cost_func,
            self.cost_init_func,
            self.penalty_,
            self.min_segment_length,
        )
        # Store the scores for introspection without recomputing using score_transform
        self.scores = pd.Series(opt_costs, index=X.index, name="score")
        return ChangeDetector._format_sparse_output(changepoints)

    def _score_transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Compute the pelt scores for the input data.

        Parameters
        ----------
        X : pd.DataFrame - data to compute scores for, time series

        Returns
        -------
        scores : pd.Series - scores for sequence `X`

        Notes
        -----
        The PELT scores are the cumulative optimal costs, so the scores are increasing
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
            {"cost": "mean", "min_segment_length": 5},
            {"cost": "mean", "penalty_scale": 0.0, "min_segment_length": 1},
        ]
        return params
