"""The pruned exact linear time (PELT) algorithm."""

__author__ = ["Tveten", "johannvk"]
__all__ = ["PELT"]

import numpy as np
import pandas as pd

from ..base import BaseIntervalScorer
from ..costs import L2Cost
from ..penalties import make_bic_penalty
from ..utils.numba import njit
from ..utils.validation.data import check_data
from ..utils.validation.parameters import check_larger_than
from .base import BaseChangeDetector


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
    cost: BaseIntervalScorer,
    penalty: float,
    min_segment_length: int,
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
    min_segment_length : int
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
    cost.check_is_fitted()
    n_samples = cost._X.shape[0]
    min_segment_shift = min_segment_length - 1

    # Explicitly set the first element to -penalty.
    opt_cost = np.concatenate((np.array([-penalty]), np.zeros(n_samples)))

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
    prev_cpts = np.repeat(0, n_samples)

    # Evolving set of admissible segment starts.
    cost_eval_starts = np.array(([0]), dtype=np.int64)

    observation_indices = np.arange(2 * min_segment_length - 1, n_samples).reshape(
        -1, 1
    )
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


class PELT(BaseChangeDetector):
    """Pruned exact linear time changepoint detection.

    The PELT algorithm [1]_ for changepoint detection.

    Parameters
    ----------
    cost : BaseIntervalScorer, optional, default=`L2Cost`
        The cost to use for the changepoint detection.
    penalty : float, optional
        The penalty to use for the changepoint detection. It must be non-negative. If
        `None`, the penalty is set to
        `make_bic_penalty(n=X.shape[0], n_params=cost.get_param_size(X.shape[1]))`,
        where ``X`` is the input data to `predict`.
    min_segment_length : int, optional, default=2
        Minimum length of a segment.
    split_cost : float, optional, default=0.0
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the above inequality.

    References
    ----------
    .. [1] Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of
    changepoints with a linear computational cost. Journal of the American Statistical
    Association, 107(500), 1590-1598.

    Examples
    --------
    >>> from skchange.change_detectors import PELT
    >>> from skchange.datasets import generate_alternating_data
    >>> df = generate_alternating_data(n_segments=2, mean=10, segment_length=100, p=5)
    >>> detector = PELT()
    >>> detector.fit_predict(df)
       ilocs
    0    100
    """

    _tags = {
        "authors": ["Tveten", "johannvk"],
        "maintainers": ["Tveten", "johannvk"],
        "fit_is_empty": True,
    }

    def __init__(
        self,
        cost: BaseIntervalScorer = None,
        penalty: float | None = None,
        min_segment_length: int = 2,
        split_cost: float = 0.0,
    ):
        self.cost = cost
        self.penalty = penalty
        self.min_segment_length = min_segment_length
        self.split_cost = split_cost
        super().__init__()

        self._cost = L2Cost() if cost is None else cost.clone()
        if self._cost.is_penalised_score:
            raise ValueError("PELT does not support inherently penalised costs.")

        check_larger_than(0, penalty, "penalty", allow_none=True)
        check_larger_than(1, min_segment_length, "min_segment_length")

        self.set_tags(distribution_type=self._cost.get_tag("distribution_type"))

    def _predict(self, X: pd.DataFrame | pd.Series) -> pd.Series:
        """Detect events in test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series to detect change points in.

        Returns
        -------
        y_sparse : pd.DataFrame
            A `pd.DataFrame` with a range index and one column:
            * ``"ilocs"`` - integer locations of the changepoints.

        Attributes
        ----------
        fitted_cost : BaseIntervalScorer
            The fitted cost function.
        fitted_penalty : float
            The fitted penalty value. Either the user-specified value or the used BIC
            penalty.
        """
        X = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="2*min_segment_length",
        )

        self.fitted_cost: BaseIntervalScorer = self._cost.clone()
        self.fitted_cost.fit(X)

        if self.penalty is None:
            self.fitted_penalty = make_bic_penalty(
                n=X.shape[0],
                n_params=self.fitted_cost.get_param_size(X.shape[1]),
            )
        else:
            self.fitted_penalty = self.penalty

        opt_costs, changepoints = run_pelt(
            cost=self.fitted_cost,
            penalty=self.fitted_penalty,
            min_segment_length=self.min_segment_length,
            split_cost=self.split_cost,
        )
        # Store the scores for introspection without recomputing using transform_scores
        self.scores = pd.Series(opt_costs, index=X.index, name="score")
        return self._format_sparse_output(changepoints)

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
            {"cost": L2Cost(), "min_segment_length": 5},
            {"cost": L2Cost(), "penalty": 0.0, "min_segment_length": 1},
        ]
        return params
