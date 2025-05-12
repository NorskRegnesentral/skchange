"""The pruned exact linear time (PELT) algorithm."""

__author__ = ["Tveten", "johannvk"]
__all__ = ["PELT"]

from functools import reduce

import numpy as np
import pandas as pd

from ..base import BaseIntervalScorer
from ..costs import L2Cost
from ..costs.base import BaseCost
from ..penalties import make_bic_penalty
from ..utils.numba import njit
from ..utils.validation.data import check_data
from ..utils.validation.interval_scorer import check_interval_scorer
from ..utils.validation.parameters import check_larger_than_or_equal
from ..utils.validation.penalties import check_penalty
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
    cost: BaseCost,
    penalty: float,
    min_segment_length: int,
    split_cost: float = 0.0,
    percent_pruning_margin: float = 0.0,
    restricted_pruning: bool = True,
    drop_pruning: bool = False,
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
        The percentage of pruning margin to use. By default set to 10.0.
        This is used to prune the admissible starts set.
        The pruning margin is used to avoid numerical issues when comparing
        the candidate optimal costs with the current optimal cost.
    restricted_pruning : bool, optional
        If True, do not prune indices from the potential changepoints set
        if they're within `2 * min_segment_length` of the current observation.
    drop_pruning: bool, optional
        If True, drop the pruning step. Reverts to optimal partitioning.
        Can be useful for debugging and testing.  By default set to False.

    Returns
    -------
    tuple[np.ndarray, list]
        The optimal costs and the changepoints.
    """
    cost.check_is_fitted()
    # n_samples = cost._X.shape[0]
    n_samples = cost.n_samples()
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

        if drop_pruning:
            continue
        else:
            # Trimming the admissible starts set: (reuse the array of optimal costs)
            current_obs_ind_opt_cost = opt_cost[opt_cost_obs_ind]

            abs_current_obs_opt_cost = np.abs(current_obs_ind_opt_cost)
            start_inclusion_threshold = (
                (
                    current_obs_ind_opt_cost
                    + abs_current_obs_opt_cost * (percent_pruning_margin / 100.0)
                )
                # Moved from 'negative' on left side
                # to 'positive' on right side.
                + penalty
                # Remove from right side of inequality.
                - split_cost
            )

            start_inclusion_mask = candidate_opt_costs <= start_inclusion_threshold
            if restricted_pruning:
                # Only prune starts at least 2*min_segment_length
                # before current observation:
                start_inclusion_mask = start_inclusion_mask | (
                    cost_eval_starts >= (latest_start - min_segment_length)
                )

            cost_eval_starts = cost_eval_starts[start_inclusion_mask]

    return opt_cost[1:], get_changepoints(prev_cpts)


def run_pelt_with_jump(
    cost: BaseCost,
    penalty: float,
    jump_step: int,
    split_cost: float = 0.0,
    drop_pruning: bool = False,
) -> tuple[np.ndarray, list]:
    """Run the PELT algorithm.

    Solves the associated (compressed data) optimization problem exactly.

    Parameters
    ----------
    cost: BaseCost
        The cost to use.
    penalty : float
        The penalty incurred for adding a changepoint.
    jump_step : int
        Only indices that are multiples of `jump_step` from the start (index `0`) are
        considered as potential changepoints. This also means that the minimum segment
        length is naturally `jump_step`.
    split_cost : float, optional
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the
        above inequality.
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

    # Redefine Opt_cost[0] to start at 0.0, as done in 2014 PELT.
    # opt_cost = np.concatenate((np.array([0.0]), np.zeros(n_samples)))
    opt_cost = np.zeros(1 + n_samples)

    # Store the previous changepoint for each latest start added.
    # Used to get the final set of changepoints after the loop.
    # Initialize to invalid previous changepoints:
    prev_cpts = np.zeros(n_samples, dtype=np.int64) + n_samples

    # Evolving set of admissible segment starts.
    cost_eval_starts = np.array([], dtype=np.int64)

    observation_interval_starts = np.arange(
        start=0, stop=n_samples - jump_step + 1, step=jump_step
    )
    observation_interval_ends = np.concatenate(
        (
            np.arange(start=jump_step - 1, stop=n_samples - jump_step, step=jump_step),
            np.array([n_samples - 1]),
        )
    )
    observation_intervals = np.column_stack(
        (observation_interval_starts, observation_interval_ends)
    )

    for obs_interval_start, obs_interval_end in observation_intervals:
        # Add the next start to the admissible starts set:
        cost_eval_starts = np.concatenate(
            (cost_eval_starts, np.array([obs_interval_start]))
        )
        cost_eval_ends = np.repeat(obs_interval_end + 1, len(cost_eval_starts))
        cost_eval_intervals = np.column_stack((cost_eval_starts, cost_eval_ends))
        costs = cost.evaluate(cost_eval_intervals)
        agg_costs = np.sum(costs, axis=1)

        # Add the penalty for a new segment:
        candidate_opt_costs = opt_cost[cost_eval_starts] + agg_costs + penalty

        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        opt_cost[obs_interval_start + 1 : obs_interval_end + 1 + 1] = (
            candidate_opt_costs[argmin_candidate_cost]
        )
        prev_cpts[obs_interval_start : obs_interval_end + 1] = cost_eval_starts[
            argmin_candidate_cost
        ]

        if drop_pruning:
            continue
        else:
            # Trimming the admissible starts set: (reuse the array of optimal costs)
            current_obs_ind_opt_cost = opt_cost[obs_interval_start + 1]

            start_inclusion_threshold = (
                current_obs_ind_opt_cost
                + penalty  # Moved from 'negative' left side to 'positive' right side.
                - split_cost  # Remove from right side of inequality.
            )

            new_start_inclusion_mask = candidate_opt_costs <= start_inclusion_threshold
            cost_eval_starts = cost_eval_starts[new_start_inclusion_mask]

    return opt_cost[1:], get_changepoints(prev_cpts)


def run_restricted_optimal_partitioning(
    cost: BaseCost,
    penalty: float,
    min_segment_length: int,
    admissable_cpts: np.ndarray | set[int],
) -> tuple[np.ndarray, list]:
    """Run optimal partitioning algorithm, restricted to a set of admissable starts.

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
    admissable_starts : np.ndarray or set[int]
        The admissable starts for the segments.

    Returns
    -------
    tuple[np.ndarray, list]
        The optimal costs and the changepoints.
    """
    cost.check_is_fitted()
    n_samples = cost.n_samples()
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

    for current_obs_ind in observation_indices:
        latest_start = current_obs_ind - min_segment_shift

        # Add the next start to the admissible starts set:
        if latest_start[0] in admissable_cpts:
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

    return opt_cost[1:], get_changepoints(prev_cpts)


def run_pelt_masked(
    cost: BaseCost,
    penalty: float,
    min_segment_length: int,
    split_cost: float = 0.0,
    percent_pruning_margin: float = 0.0,
    drop_pruning: bool = False,
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

        if drop_pruning:
            continue
        else:
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
    cost : BaseIntervalScorer, optional, default=`L2Cost`
        The cost to use for the changepoint detection.
    penalty : float, optional
        The penalty to use for the changepoint detection. It must be non-negative. If
        `None`, the penalty is set to
        `make_bic_penalty(n=X.shape[0], n_params=cost.get_model_size(X.shape[1]))`,
        where ``X`` is the input data to `predict`.
    min_segment_length : int, optional, default=2
        Minimum length of a segment.
    jump: bool, optional, default=False
        If True, only indices that are multiples of `min_segment_length` from the
        first data point (index `0`) are considered as potential changepoints.
        Only used if `min_segment_length >= 2`.
    split_cost : float, optional, default=0.0
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the above inequality.
    percent_pruning_margin : float, optional, default=0.0
        The percentage of pruning margin to use. By default set to 0.0.
    drop_pruning : bool, optional, default=False
        If True, drop the pruning step. Reverts to optimal partitioning.
        Can be useful for debugging and testing. By default set to False.

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
        min_segment_length: int = 1,
        jump: bool = False,
        split_cost: float = 0.0,
        percent_pruning_margin: float = 0.0,
        drop_pruning: bool = False,
    ):
        self.cost = cost
        self.penalty = penalty
        self.min_segment_length = min_segment_length
        self.jump = jump
        self.split_cost = split_cost
        self.percent_pruning_margin = percent_pruning_margin
        self.drop_pruning = drop_pruning
        super().__init__()

        _cost = L2Cost() if cost is None else cost
        check_interval_scorer(
            _cost,
            arg_name="cost",
            caller_name="PELT",
            required_tasks=["cost"],
            allow_penalised=False,
        )
        self._cost = _cost.clone()
        self.fitted_cost: BaseIntervalScorer | None = None
        self.fitted_penalty: float | None = None

        check_penalty(
            penalty,
            "penalty",
            "PELT",
            require_constant_penalty=True,
            allow_none=True,
        )
        check_larger_than_or_equal(1, min_segment_length, "min_segment_length")

        self.clone_tags(self._cost, ["distribution_type"])

    def fit_cost_and_penalty(
        self,
        X: pd.DataFrame | pd.Series,
    ):
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
                n_params=self.fitted_cost.get_model_size(X.shape[1]),
            )
        else:
            self.fitted_penalty = self.penalty

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
            The fitted penalty value. Either the user-specified value or the fitted BIC
            penalty.
        """
        self.fit_cost_and_penalty(X)

        if self.jump and self.min_segment_length >= 2:
            opt_costs, changepoints = run_pelt_with_jump(
                cost=self.fitted_cost,
                penalty=self.fitted_penalty,
                jump_step=self.min_segment_length,
                split_cost=self.split_cost,
                drop_pruning=self.drop_pruning,
            )
        else:
            opt_costs, changepoints = run_pelt(
                cost=self.fitted_cost,
                penalty=self.fitted_penalty,
                min_segment_length=self.min_segment_length,
                split_cost=self.split_cost,
                percent_pruning_margin=self.percent_pruning_margin,
                drop_pruning=self.drop_pruning,
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

    def refine_change_points(
        self,
        change_points: list[int] | np.ndarray[int],
        X: pd.DataFrame | pd.Series | None = None,
        change_point_margin: int | None = None,
    ) -> np.ndarray[int]:
        """Refine the changepoints using the optimal partitioning algorithm.

        Parameters
        ----------
        cpts : list[int] | np.ndarray[int]
            List of changepoints to refine.
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to score (time series).
        change_point_margin : int, optional
            The margin size on each side of the provided change point, to use as
            potential change point during the refinement. If None, the default is
            `self.min_segment_length`.

        Returns
        -------
        refined_cpts : np.ndarray[int]
            List of refined changepoints.
        """
        self.check_is_fitted()

        if X is not None:
            self.fit_cost_and_penalty(X)
        else:
            if self.fitted_cost is None or self.fitted_penalty is None:
                raise RuntimeError(
                    "The `PELT` cost and penalty have not been fitted yet. "
                    "Please call `.predict()` before refining changepoints."
                )

        if change_point_margin is None:
            change_point_margin = self.min_segment_length

        if len(change_points) == 0:
            return np.array([], dtype=np.int64)
        else:
            # Construct the admissible change points set:
            admissable_cpts = reduce(
                lambda x, y: x | y,
                [
                    set(
                        range(
                            cpt - (change_point_margin - 1),
                            cpt + (change_point_margin - 1) + 1,
                        )
                    )
                    for cpt in change_points
                ],
            )

            refined_cpts = run_restricted_optimal_partitioning(
                cost=self.fitted_cost,
                penalty=self.fitted_penalty,
                min_segment_length=self.min_segment_length,
                admissable_cpts=admissable_cpts,
            )[1]

            return refined_cpts

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
