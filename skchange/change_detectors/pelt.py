"""The pruned exact linear time (PELT) algorithm."""

__author__ = ["Tveten", "johannvk"]
__all__ = ["PELT"]


from typing import Optional, Union

import numpy as np
import pandas as pd

from skchange.change_detectors.base import ChangeDetector
from skchange.costs import BaseCost, L2Cost
from skchange.utils.numba import njit
from skchange.utils.validation.data import check_data
from skchange.utils.validation.parameters import check_larger_than


@njit
def get_changepoints(prev_cpts: np.ndarray) -> np.ndarray:
    changepoints = []
    i = len(prev_cpts) - 1
    while i >= 0:
        cpt_i = prev_cpts[i]
        changepoints.append(cpt_i)
        i = cpt_i - 1
    return np.array(changepoints[-2::-1])  # Remove the artificial changepoint at 0.


def run_pelt(
    X: np.ndarray,
    cost: BaseCost,
    penalty,
    min_segment_length: int = 1,
    split_cost: float = 0.0,
) -> tuple[np.ndarray, list]:
    """Run the PELT algorithm.

    Currently agrees with the 'changepoint::cpt.mean' implementation of PELT in R.
    If the 'min_segment_length' is large enough to span more than a single changepoint,
    the algorithm can return a suboptimal partitioning.
    In that case, resort to the 'optimal_partitioning' algorithm.

    Parameters
    ----------
    X : np.ndarray
        The data to find changepoints in.
    cost: BaseCost
        The cost to use.
    penalty : float
        The penalty incurred for adding a changepoint.
    min_segment_length : int, optional
        The minimum length of a segment, by default 1.
    split_cost : float, optional
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the
        above inequality.

    Returns
    -------
    tuple[np.ndarray, list]
        The optimal costs and the changepoints.
    """
    num_obs = len(X)
    cost.fit(X)
    min_segment_shift = min_segment_length - 1

    # Explicitly set the first element to -penalty.
    opt_cost = np.concatenate((np.array([-penalty]), np.zeros(num_obs)))

    # Cannot compute the cost for the first 'min_segment_shift' elements:
    opt_cost[1:min_segment_length] = -penalty

    # Compute the cost in [min_segment_length, 2*min_segment_length - 1] directly:
    non_changepoint_starts = np.zeros(min_segment_length, dtype=np.int64)
    non_changepoint_ends = np.arange(min_segment_length, 2 * min_segment_length)
    non_changepoint_intervals = np.column_stack(
        (non_changepoint_starts, non_changepoint_ends)
    )
    costs = cost.evaluate(non_changepoint_intervals)
    agg_costs = np.sum(costs, axis=1)
    opt_cost[min_segment_length : 2 * min_segment_length] = agg_costs

    # Store the previous changepoint for each latest start added.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = np.repeat(0, num_obs)

    # Evolving set of admissible segment starts.
    cost_eval_starts = np.array(([0]), dtype=np.int64)

    observation_indices = np.arange(2 * min_segment_length - 1, num_obs).reshape(-1, 1)
    for current_obs_ind in observation_indices:
        latest_start = current_obs_ind - min_segment_shift

        # Add the next start to the admissible starts set:
        cost_eval_starts = np.concatenate((cost_eval_starts, latest_start))
        cost_eval_ends = np.repeat(current_obs_ind + 1, len(cost_eval_starts))
        cost_eval_intervals = np.column_stack((cost_eval_starts, cost_eval_ends))
        costs = cost.evaluate(cost_eval_intervals)
        agg_costs = np.sum(costs, axis=1)

        candidate_opt_costs = opt_cost[cost_eval_starts] + agg_costs + penalty

        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        opt_cost[current_obs_ind + 1] = candidate_opt_costs[argmin_candidate_cost]
        prev_cpts[current_obs_ind] = cost_eval_starts[argmin_candidate_cost]

        # Trimming the admissible starts set: (reuse the array of optimal costs)
        cost_eval_starts = cost_eval_starts[
            candidate_opt_costs + split_cost <= opt_cost[current_obs_ind + 1] + penalty
        ]

    return opt_cost[1:], get_changepoints(prev_cpts)


class PELT(ChangeDetector):
    """Pruned exact linear time changepoint detection.

    An efficient implementation of the PELT algorithm [1]_ for changepoint detection.

    Parameters
    ----------
    cost : BaseCost, optional (default=`L2Cost`)
        The cost function to use for the changepoint detection.
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
    >>> from skchange.change_detectors import PELT
    >>> from skchange.datasets.generate import generate_alternating_data
    >>> df = generate_alternating_data(n_segments=2, mean=10, segment_length=100, p=5)
    >>> detector = PELT()
    >>> detector.fit_predict(df)
    0    99
    Name: changepoint, dtype: int64
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        cost: BaseCost = L2Cost(),
        penalty_scale: Optional[float] = 2.0,
        min_segment_length: int = 2,
    ):
        self.cost = cost
        self.penalty_scale = penalty_scale
        self.min_segment_length = min_segment_length
        super().__init__()

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
            self.cost,
            self.penalty_,
            self.min_segment_length,
        )
        # Store the scores for introspection without recomputing using transform_scores
        self.scores = pd.Series(opt_costs, index=X.index, name="score")
        return ChangeDetector._format_sparse_output(changepoints)

    def _transform_scores(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
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
        from skchange.costs import L2Cost

        params = [
            {"cost": L2Cost(), "min_segment_length": 5},
            {"cost": L2Cost(), "penalty_scale": 0.0, "min_segment_length": 1},
        ]
        return params
