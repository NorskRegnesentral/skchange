"""The pruned exact linear time (PELT) algorithm."""

__author__ = ["Tveten", "johannvk"]
__all__ = ["PELT"]

import numpy as np
import pandas as pd

from ..compose.penalised_score import PenalisedScore
from ..costs import L2Cost
from ..costs.base import BaseCost
from ..penalties import BICPenalty, as_penalty
from ..penalties.base import BasePenalty
from ..utils.numba import njit
from ..utils.validation.data import check_data
from ..utils.validation.parameters import check_larger_than_or_equal
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


def run_improved_pelt_array_based(
    cost: BaseCost,
    penalty: float,
    min_segment_length: int,
    split_cost: float = 0.0,
    percent_pruning_margin: float = 0.0,
    drop_pruning: bool = False,
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
    percent_pruning_margin : float, optional
        The percentage of pruning margin to use. By default set to 10.0.
        This is used to prune the admissible starts set.
        The pruning margin is used to avoid numerical issues when comparing
        the candidate optimal costs with the current optimal cost.
    drop_pruning: bool, optional
        If True, drop the pruning step. Reverts to optimal partitioning.
        Can be useful for debugging and testing.  By default set to False.

    Returns
    -------
    tuple[np.ndarray, list]
        The optimal costs and the changepoints.
    """
    cost.check_is_fitted()
    n_samples = cost._X.shape[0]
    min_segment_shift = min_segment_length - 1

    # Redefine Opt_cost[0] to start at 0.0, as done in 2014 PELT.
    opt_cost = np.concatenate((np.array([0.0]), np.zeros(n_samples)))

    # Cannot compute the cost for the first 'min_segment_shift' elements:
    opt_cost[1:min_segment_length] = np.inf

    # Compute the cost in [min_segment_length, 2*min_segment_length - 1] directly:
    non_changepoint_starts = np.zeros(min_segment_length, dtype=np.int64)
    non_changepoint_ends = np.arange(min_segment_length, 2 * min_segment_length)
    non_changepoint_intervals = np.column_stack(
        (non_changepoint_starts, non_changepoint_ends)
    )
    costs = cost.evaluate(non_changepoint_intervals)
    agg_costs = np.sum(costs, axis=1)
    opt_cost[min_segment_length : 2 * min_segment_length] = agg_costs + penalty

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
        opt_cost_obs_ind = current_obs_ind[0] + 1

        # Add the next start to the admissible starts set:
        cost_eval_starts = np.concatenate((cost_eval_starts, latest_start))
        cost_eval_ends = np.repeat(current_obs_ind + 1, len(cost_eval_starts))
        cost_eval_intervals = np.column_stack((cost_eval_starts, cost_eval_ends))
        costs = cost.evaluate(cost_eval_intervals)
        agg_costs = np.sum(costs, axis=1)

        # Add the penalty for a new segment:
        candidate_opt_costs = opt_cost[cost_eval_starts] + agg_costs + penalty

        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        opt_cost[opt_cost_obs_ind] = candidate_opt_costs[argmin_candidate_cost]
        prev_cpts[current_obs_ind] = cost_eval_starts[argmin_candidate_cost]

        # Trimming the admissible starts set: (reuse the array of optimal costs)
        current_obs_ind_opt_cost = opt_cost[opt_cost_obs_ind]

        abs_current_obs_opt_cost = np.abs(current_obs_ind_opt_cost)
        start_inclusion_threshold = (
            (
                current_obs_ind_opt_cost
                + abs_current_obs_opt_cost * (percent_pruning_margin / 100.0)
            )
            + penalty  # Moved from 'negative' on left side to 'positive' on right side.
            - split_cost  # Remove from right side of inequality.
        )

        # Only prune starts at least 2*min_segment_length before current observation:
        new_start_inclusion_mask = (
            candidate_opt_costs <= start_inclusion_threshold
        ) | (cost_eval_starts >= latest_start - min_segment_length)
        # Two 'off by one' cases?
        # ) | (cost_eval_starts >= latest_start - min_segment_length - 2)

        if not drop_pruning:
            cost_eval_starts = cost_eval_starts[
                # Introduce a small tolerance to avoid numerical issues:
                # candidate_opt_costs + split_cost <= start_inclusion_threshold
                # old_start_inclusion_mask
                new_start_inclusion_mask
            ]

    return opt_cost[1:], get_changepoints(prev_cpts)


def run_pelt_array_based(
    cost: BaseCost,
    penalty: float,
    min_segment_length: int,
    split_cost: float = 0.0,
    percent_pruning_margin: float = 0.0,
    drop_pruning: bool = False,
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
    percent_pruning_margin : float, optional
        The percentage of pruning margin to use. By default set to 10.0.
        This is used to prune the admissible starts set.
        The pruning margin is used to avoid numerical issues when comparing
        the candidate optimal costs with the current optimal cost.
    drop_pruning: bool, optional
        If True, drop the pruning step. Reverts to optimal partitioning.
        Can be useful for debugging and testing.  By default set to False.

    Returns
    -------
    tuple[np.ndarray, list]
        The optimal costs and the changepoints.
    """
    cost.check_is_fitted()
    n_samples = cost._X.shape[0]
    min_segment_shift = min_segment_length - 1

    # Redefine Opt_cost[0] to start at 0.0, as done in 2014 PELT.
    opt_cost = np.concatenate((np.array([0.0]), np.zeros(n_samples)))

    # Cannot compute the cost for the first 'min_segment_shift' elements:
    opt_cost[1:min_segment_length] = 0.0

    # Compute the cost in [min_segment_length, 2*min_segment_length - 1] directly:
    non_changepoint_starts = np.zeros(min_segment_length, dtype=np.int64)
    non_changepoint_ends = np.arange(min_segment_length, 2 * min_segment_length)
    non_changepoint_intervals = np.column_stack(
        (non_changepoint_starts, non_changepoint_ends)
    )
    costs = cost.evaluate(non_changepoint_intervals)
    agg_costs = np.sum(costs, axis=1)
    opt_cost[min_segment_length : 2 * min_segment_length] = agg_costs + penalty

    # Store the previous changepoint for each latest start added.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = np.repeat(0, n_samples)

    # Evolving set of admissible segment starts.
    cost_eval_starts = np.array(([0]), dtype=np.int64)

    observation_indices = np.arange(2 * min_segment_length - 1, n_samples).reshape(
        -1, 1
    )

    # for current_obs_ind in range(2 * min_segment_length - 1, n_samples):
    for current_obs_ind in observation_indices:
        latest_start = current_obs_ind - min_segment_shift

        # Add the next start to the admissible starts set:
        cost_eval_starts = np.concatenate((cost_eval_starts, latest_start))
        cost_eval_ends = np.repeat(current_obs_ind + 1, len(cost_eval_starts))
        cost_eval_intervals = np.column_stack((cost_eval_starts, cost_eval_ends))
        costs = cost.evaluate(cost_eval_intervals)
        agg_costs = np.sum(costs, axis=1)

        # Add the penalty for a new segment:
        candidate_opt_costs = opt_cost[cost_eval_starts] + agg_costs + penalty

        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        opt_cost[current_obs_ind + 1] = candidate_opt_costs[argmin_candidate_cost]
        prev_cpts[current_obs_ind] = cost_eval_starts[argmin_candidate_cost]

        # Trimming the admissible starts set: (reuse the array of optimal costs)
        current_obs_ind_opt_cost = opt_cost[current_obs_ind + 1]
        # Handle cases where the optimal cost is negative:
        abs_current_obs_opt_cost = np.abs(current_obs_ind_opt_cost)
        start_inclusion_threshold = (
            (
                current_obs_ind_opt_cost
                + abs_current_obs_opt_cost * (percent_pruning_margin / 100.0)
            )
            + penalty  # Moved from 'negative' on left side to 'positive' on right side.
            - split_cost  # Remove from right side of inequality.
        )

        if not drop_pruning:
            cost_eval_starts = cost_eval_starts[
                # Introduce a small tolerance to avoid numerical issues:
                candidate_opt_costs <= start_inclusion_threshold
            ]

    return opt_cost[1:], get_changepoints(prev_cpts)


def run_pelt_masked(
    cost: BaseCost,
    penalty: float,
    min_segment_length: int,
    split_cost: float = 0.0,
    percent_pruning_margin: float = 0.1,
    pre_allocation_multiplier: float = 5.0,  # Initial multiple of log(n_samples)
    growth_factor: float = 2.0,  # Geometric growth factor
) -> tuple[np.ndarray, list]:
    """Run the PELT algorithm.

    Currently agrees with the 'changepoint::cpt.mean' implementation of PELT in R.
    If the 'min_segment_length' is large enough to span more than a single changepoint,
    the algorithm can return a suboptimal partitioning.
    In that case, resort to the 'optimal_partitioning' algorithm.

    Parameters
    ----------
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
    percent_pruning_margin : float, optional
        The percentage of pruning margin to use. By default set to 0.1.
        This is used to prune the admissible starts set.
        The pruning margin is used to avoid numerical issues when comparing
        the candidate optimal costs with the current optimal cost.
    initial_capacity : float, optional
        The initial capacity of the arrays as a fraction of n_samples.
        Default is 0.1 (10% of n_samples).
    growth_factor : float, optional
        The factor by which to grow the arrays when they need to be resized.
        Default is 2.0.

    Returns
    -------
    tuple[np.ndarray, list]
        The optimal costs and the changepoints.
    """
    cost.check_is_fitted()
    n_samples = cost._X.shape[0]
    min_segment_shift = min_segment_length - 1

    # Explicitly set the first element to 0.
    # Define "opt_cost[0]"" to start at 0.0, as done in 2014 PELT.
    opt_cost = np.concatenate((np.array([0.0]), np.zeros(n_samples)))

    # Cannot compute the cost for the first 'min_segment_shift' elements:
    opt_cost[1:min_segment_length] = 0.0

    # Compute the cost in [min_segment_length, 2*min_segment_length - 1] directly:
    non_changepoint_starts = np.zeros(min_segment_length, dtype=np.int64)
    non_changepoint_ends = np.arange(min_segment_length, 2 * min_segment_length)
    non_changepoint_intervals = np.column_stack(
        (non_changepoint_starts, non_changepoint_ends)
    )
    costs = cost.evaluate(non_changepoint_intervals)
    agg_costs = np.sum(costs, axis=1)
    opt_cost[min_segment_length : 2 * min_segment_length] = agg_costs + penalty

    # Store the previous changepoint for each latest start added.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = np.repeat(0, n_samples)

    # Initialize smaller arrays with a multiple of log(n_samples) capacity:
    initial_allocation_size = max(2, int(np.log(n_samples) * pre_allocation_multiplier))

    # Pre-allocate arrays with initial capacity
    starts_capacity = initial_allocation_size
    starts_buffer = np.zeros(starts_capacity, dtype=np.int64)
    interval_capacity = initial_allocation_size
    interval_buffer = np.zeros((interval_capacity, 2), dtype=np.int64)

    # Initialize with the first valid start position (position 0)
    n_valid_starts = 1
    starts_buffer[0] = 0  # First valid start is at position 0

    for current_obs_ind in range(2 * min_segment_length - 1, n_samples):
        latest_start = current_obs_ind - min_segment_shift

        # Add the next start position to the admissible set:
        # First check if we need to grow the arrays
        if n_valid_starts + 1 > starts_capacity:
            # Grow arrays geometrically
            new_capacity = int(starts_capacity * growth_factor)
            new_starts_buffer = np.zeros(new_capacity, dtype=np.int64)
            new_starts_buffer[:n_valid_starts] = starts_buffer[:n_valid_starts]
            starts_buffer = new_starts_buffer
            starts_capacity = new_capacity

            # Also grow the interval buffer
            new_interval_capacity = int(interval_capacity * growth_factor)
            new_interval_buffer = np.zeros((new_interval_capacity, 2), dtype=np.int64)
            new_interval_buffer[:interval_capacity] = interval_buffer[
                :interval_capacity
            ]
            interval_buffer = new_interval_buffer
            interval_capacity = new_interval_capacity

        # Add the latest start to the buffer of valid starts
        starts_buffer[n_valid_starts] = latest_start
        n_valid_starts += 1

        # Set up intervals for cost evaluation
        current_end = current_obs_ind + 1

        # Fill the interval buffer with current valid starts and the current end
        interval_buffer[:n_valid_starts, 0] = starts_buffer[:n_valid_starts]
        interval_buffer[:n_valid_starts, 1] = current_end

        # Evaluate costs:
        agg_costs = np.sum(cost.evaluate(interval_buffer[:n_valid_starts]), axis=1)

        # Add the cost and penalty for a new segment (since last changepoint)
        # Reusing the agg_costs array to store the candidate optimal costs.
        agg_costs[:] += penalty + opt_cost[starts_buffer[:n_valid_starts]]
        candidate_opt_costs = agg_costs

        # Find the optimal cost and previous changepoint:
        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        min_start_idx = starts_buffer[argmin_candidate_cost]
        opt_cost[current_obs_ind + 1] = candidate_opt_costs[argmin_candidate_cost]
        prev_cpts[current_obs_ind] = min_start_idx

        # Pruning: update valid starts to exclude positions that cannot be optimal
        current_obs_ind_opt_cost = opt_cost[current_obs_ind + 1]
        abs_current_obs_opt_cost = np.abs(current_obs_ind_opt_cost)

        # Calculate pruning threshold with margin:
        start_inclusion_threshold = (
            (
                current_obs_ind_opt_cost
                + abs_current_obs_opt_cost * (percent_pruning_margin / 100.0)
            )
            + penalty  # Pruning inequality does not include added penalty.
            - split_cost  # Remove from right side of inequality.
        )

        # Apply pruning by filtering valid starts:
        valid_starts_mask = (
            candidate_opt_costs[:n_valid_starts] <= start_inclusion_threshold
        )
        n_new_valid_starts = np.sum(valid_starts_mask)
        starts_buffer[:n_new_valid_starts] = starts_buffer[:n_valid_starts][
            valid_starts_mask
        ]
        n_valid_starts = n_new_valid_starts

    return opt_cost[1:], get_changepoints(prev_cpts)


class PELT(BaseChangeDetector):
    """Pruned exact linear time changepoint detection.

    The PELT algorithm [1]_ for changepoint detection.

    Parameters
    ----------
    cost : BaseCost, optional, default=`L2Cost`
        The cost function to use for the changepoint detection.
    penalty : BasePenalty or float, optional, default=`BICPenalty`
        The penalty to use for the changepoint detection. If a float is given, it is
        interpreted as a constant penalty. If `None`, the penalty is set to a BIC
        penalty with ``n=X.shape[0]`` and ``n_params=cost.get_param_size(X.shape[1])``,
        where ``X`` is the input data to `predict`.
    min_segment_length : int, optional, default=2
        Minimum length of a segment.
    split_cost : float, optional, default=0.0
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the above inequality.
    percent_pruning_margin : float, optional, default=0.0
        The percentage of pruning margin to use. By default set to 0.0.

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
        cost: BaseCost = None,
        penalty: BasePenalty | float | None = None,
        min_segment_length: int = 1,
        split_cost: float = 0.0,
        percent_pruning_margin: float = 0.0,
    ):
        self.cost = cost
        self.penalty = penalty
        self.min_segment_length = min_segment_length
        self.split_cost = split_cost
        self.percent_pruning_margin = percent_pruning_margin
        super().__init__()

        self._cost = L2Cost() if cost is None else cost
        self._penalty = as_penalty(
            self.penalty, default=BICPenalty(), require_penalty_type="constant"
        )
        self._penalised_cost = (
            self._cost.clone()  # need to avoid modifying the input cost
            if self._cost.is_penalised_score
            else PenalisedScore(self._cost, self._penalty)
        )
        check_larger_than_or_equal(1, min_segment_length, "min_segment_length")

    def update_penalty(self, penalty: float | BasePenalty) -> None:
        """Update the penalty of the cost function.

        Parameters
        ----------
        penalty : float or BasePenalty
            The new penalty to use.
        """
        self.penalty = penalty
        self._penalty = as_penalty(self.penalty, require_penalty_type="constant")
        self._penalised_cost = (
            self._cost.clone()  # need to avoid modifying the input cost
            if self._cost.is_penalised_score
            else PenalisedScore(self._cost, self._penalty)
        )

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
        """
        X = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="2*min_segment_length",
        )
        self._penalised_cost.fit(X)
        # opt_costs, changepoints = run_pelt_array_based(
        opt_costs, changepoints = run_improved_pelt_array_based(
            cost=self._penalised_cost.scorer_,
            penalty=self._penalised_cost.penalty_.values[0],
            min_segment_length=self.min_segment_length,
            split_cost=self.split_cost,
            percent_pruning_margin=self.percent_pruning_margin,
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
