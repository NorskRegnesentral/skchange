"""Tests for the PELT implementation."""

import time

import numpy as np
import pandas as pd
import pytest

from skchange.change_detectors._pelt import (
    get_changepoints,
    run_pelt,
)
from skchange.costs import L2Cost
from skchange.costs.base import BaseCost
from skchange.datasets import generate_alternating_data

n_segments = 2
seg_len = 50
changepoint_data = generate_alternating_data(
    n_segments=n_segments, mean=20, segment_length=seg_len, p=1, random_state=2
).values.reshape(-1, 1)

alternating_sequence = generate_alternating_data(
    n_segments=5, mean=10.5, variance=0.5, segment_length=20, p=1, random_state=5
).values.reshape(-1, 1)


@pytest.fixture
def cost():
    """Generate a new cost object for each test."""
    cost = L2Cost()
    return cost


@pytest.fixture
def penalty() -> float:
    """Penalty for the PELT algorithm."""
    penalty = 2 * np.log(len(changepoint_data))
    return penalty


def pelt_partition_cost(
    X: np.ndarray,
    changepoints: np.ndarray,
    cost: BaseCost,
    penalty: float,
):
    cost.fit(X)
    num_samples = len(X)

    # Add number of 'segments' * penalty to the cost.
    # Instead of number of 'changepoints' * penalty.
    total_cost = penalty * (len(changepoints) + 1)
    np_changepoints = np.asarray(changepoints, dtype=np.int64)

    interval_starts = np.concatenate((np.array([0]), np_changepoints), axis=0)
    interval_ends = np.concatenate((np_changepoints, np.array([num_samples])), axis=0)

    interval_costs = np.sum(
        cost.evaluate(np.column_stack((interval_starts, interval_ends))), axis=1
    )
    total_cost += np.sum(interval_costs)

    return total_cost


def run_pelt_old(
    cost: BaseCost, penalty: float, min_segment_length: int
) -> tuple[np.ndarray, list]:
    # With 'min_segment_length' > 1, this function can return
    # segment lengths < 'min_segment_length'.
    cost.check_is_fitted()
    n_samples = cost._X.shape[0]

    starts = np.array((), dtype=np.int64)  # Evolving set of admissible segment starts.
    init_starts = np.zeros(min_segment_length - 1, dtype=np.int64)
    init_ends = np.arange(min_segment_length - 1)
    opt_cost = np.zeros(n_samples + 1) - penalty
    opt_cost[1:min_segment_length] = np.sum(
        cost.evaluate(np.column_stack((init_starts, init_ends + 1))), axis=1
    )

    # Store the previous changepoint for each t.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = np.repeat(-1, n_samples)

    ts = np.arange(min_segment_length - 1, n_samples).reshape(-1, 1)
    for t in ts:
        starts = np.concatenate((starts, t - min_segment_length + 1))
        ends = np.repeat(t, len(starts))
        candidate_opt_costs = (
            opt_cost[starts]
            + np.sum(cost.evaluate(np.column_stack((starts, ends + 1))), axis=1)
            + penalty
        )
        argmin = np.argmin(candidate_opt_costs)
        opt_cost[t + 1] = candidate_opt_costs[argmin]
        prev_cpts[t] = starts[argmin] - 1

        # Trimming the admissible starts set
        starts = starts[candidate_opt_costs - penalty <= opt_cost[t]]

    return opt_cost[1:], get_changepoints(prev_cpts)


def run_optimal_partitioning(
    cost: BaseCost,
    penalty,
    min_segment_length: int = 1,
) -> tuple[np.ndarray, list]:
    # The simpler and more direct 'optimal partitioning' algorithm,
    # as compared to the PELT algorithm.
    cost.check_is_fitted()
    n_samples = cost._X.shape[0]
    min_segment_shift = min_segment_length - 1

    # Explicitly set the first element to -penalty, and the rest to NaN.
    # Last 'min_segment_shift' elements will be NaN.
    # opt_cost = np.concatenate((np.array([-penalty]), np.zeros(n_samples)))
    opt_cost = np.concatenate((np.array([0.0]), np.zeros(n_samples)))

    # If min_segment_length > 1, cannot compute the cost for the first
    # [1, .., min_segment_length - 1] observations.
    # opt_cost[1:min_segment_length] = -penalty
    opt_cost[1:min_segment_length] = 0.0

    # Compute the optimal cost for the first
    # [min_segment_length, .., 2* min_segment_length - 1] observations
    # directly from the cost function, as we cannot have a changepoint
    # within the first [min_segment_length, 2*min_segment_length - 1]
    # observations.
    non_changepoint_starts = np.zeros(min_segment_length, dtype=np.int64)
    non_changepoint_ends = np.arange(min_segment_length - 1, 2 * min_segment_length - 1)

    # Shifted by 1 to account for the first element being -penalty:
    opt_cost[min_segment_length : (2 * min_segment_length)] = (
        np.sum(
            cost.evaluate(
                np.column_stack((non_changepoint_starts, non_changepoint_ends + 1))
            ),
            axis=1,
        )
        + penalty
    )

    # Store the previous changepoint for each last start added.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = np.repeat(-1, n_samples)

    # Evolving set of admissible segment starts.
    # Always include [0] as the start of a contiguous segment.
    candidate_starts = np.array(([0]), dtype=np.int64)

    opt_cost_observation_indices = np.arange(
        2 * min_segment_length - 1, n_samples
    ).reshape(-1, 1)

    for opt_cost_obs_index in opt_cost_observation_indices:
        segment_start = opt_cost_obs_index - min_segment_shift

        # Add the next start to the admissible starts set:
        candidate_starts = np.concatenate((candidate_starts, segment_start))
        candidate_ends = np.repeat(opt_cost_obs_index, len(candidate_starts))

        candidate_opt_costs = (
            opt_cost[candidate_starts]  # Shifted by one.
            + np.sum(
                cost.evaluate(np.column_stack((candidate_starts, candidate_ends + 1))),
                axis=1,
            )
            + penalty
        )

        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        opt_cost[opt_cost_obs_index + 1] = candidate_opt_costs[argmin_candidate_cost]
        prev_cpts[opt_cost_obs_index] = candidate_starts[argmin_candidate_cost] - 1

    return opt_cost[1:], get_changepoints(prev_cpts)


def run_pelt_array_based(
    cost: BaseCost,
    penalty: float,
    min_segment_length: int,
    split_cost: float = 0.0,
    percent_pruning_margin: float = 0.0,
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
    cost_eval_time = 0.0
    # for current_obs_ind in range(2 * min_segment_length - 1, n_samples):
    for current_obs_ind in observation_indices:
        latest_start = current_obs_ind - min_segment_shift

        # Add the next start to the admissible starts set:
        cost_eval_starts = np.concatenate((cost_eval_starts, latest_start))
        cost_eval_ends = np.repeat(current_obs_ind + 1, len(cost_eval_starts))
        cost_eval_intervals = np.column_stack((cost_eval_starts, cost_eval_ends))
        cost_eval_t0 = time.perf_counter()
        costs = cost.evaluate(cost_eval_intervals)
        agg_costs = np.sum(costs, axis=1)
        cost_eval_t1 = time.perf_counter()
        cost_eval_time += cost_eval_t1 - cost_eval_t0

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
            current_obs_ind_opt_cost
            + abs_current_obs_opt_cost * (percent_pruning_margin / 100.0)
        ) + penalty  # Moved from 'negative' on left side to 'positive' on right side.

        cost_eval_starts = cost_eval_starts[
            # Introduce a small tolerance to avoid numerical issues:
            candidate_opt_costs + split_cost <= start_inclusion_threshold
        ]

    return opt_cost[1:], get_changepoints(prev_cpts), cost_eval_time


def run_pelt_masked(
    cost: BaseCost,
    penalty: float,
    min_segment_length: int,
    split_cost: float = 0.0,
    percent_pruning_margin: float = 0.0,
    initial_capacity: float = 0.1,  # Initial fraction of n_samples
    growth_factor: float = 1.5,  # Geometric growth factor
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
    initial_capacity : float, optional
        The initial capacity of the arrays as a fraction of n_samples.
        Default is 0.1 (10% of n_samples).
    growth_factor : float, optional
        The factor by which to grow the arrays when they need to be resized.
        Default is 1.5.

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

    # Initialize smaller arrays with a fraction of n_samples capacity
    initial_size = max(2, int(n_samples * initial_capacity))

    # Pre-allocate arrays with initial capacity
    start_capacity = initial_size
    starts_buffer = np.zeros(start_capacity, dtype=np.int64)
    interval_capacity = initial_size
    interval_buffer = np.zeros((interval_capacity, 2), dtype=np.int64)

    # Initialize with the first valid start position (position 0)
    n_valid_starts = 1
    starts_buffer[0] = 0  # First valid start is at position 0
    cost_eval_time = 0.0

    for current_obs_ind in range(2 * min_segment_length - 1, n_samples):
        latest_start = current_obs_ind - min_segment_shift

        # Add the next start position to the admissible set:
        # First check if we need to grow the arrays
        if n_valid_starts + 1 > start_capacity:
            # Grow arrays geometrically
            new_capacity = int(start_capacity * growth_factor)
            new_starts_buffer = np.zeros(new_capacity, dtype=np.int64)
            new_starts_buffer[:n_valid_starts] = starts_buffer[:n_valid_starts]
            starts_buffer = new_starts_buffer
            start_capacity = new_capacity

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
        cost_eval_t0 = time.perf_counter()
        costs = cost.evaluate(interval_buffer[:n_valid_starts])
        agg_costs = np.sum(costs, axis=1)
        cost_eval_t1 = time.perf_counter()
        cost_eval_time += cost_eval_t1 - cost_eval_t0

        # Add the cost and penalty for a new segment (since last changepoint)
        candidate_opt_costs = np.zeros(n_valid_starts)
        candidate_opt_costs[:n_valid_starts] = (
            opt_cost[starts_buffer[:n_valid_starts]] + agg_costs + penalty
        )

        # Find the optimal cost and previous changepoint
        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        min_start_idx = starts_buffer[argmin_candidate_cost]
        opt_cost[current_obs_ind + 1] = candidate_opt_costs[argmin_candidate_cost]
        prev_cpts[current_obs_ind] = min_start_idx

        # Pruning: update valid starts to exclude positions that cannot be optimal
        current_obs_ind_opt_cost = opt_cost[current_obs_ind + 1]
        abs_current_obs_opt_cost = np.abs(current_obs_ind_opt_cost)

        # Calculate pruning threshold with margin
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

    return opt_cost[1:], get_changepoints(prev_cpts), cost_eval_time


def test_benchmark_pelt_implementations(cost: BaseCost, penalty: float):
    """Benchmark different PELT implementations."""

    # Generate a larger dataset for benchmarking: 10_000
    n_segments = 10
    seg_len = 1_000
    benchmark_data = generate_alternating_data(
        n_segments=n_segments,
        mean=20,
        segment_length=seg_len,
        p=1,
        # random_state=2
    ).values.reshape(-1, 1)

    cost.fit(benchmark_data)

    # Parameters to test
    min_segment_lengths = [1, 5, 10]

    # Store results:
    results = []

    for min_segment_length in min_segment_lengths:
        # Benchmark run_pelt
        start_time = time.perf_counter()
        pelt_costs_array_based, array_cpts, masked_cost_eval_time = run_pelt_masked(
            cost,
            penalty=penalty,
            min_segment_length=min_segment_length,
        )
        masked_pelt_time = time.perf_counter() - start_time
        masked_overhead = masked_pelt_time - masked_cost_eval_time

        # Benchmark run_pelt_array_based
        start_time = time.perf_counter()
        pelt_costs_masked, masked_cpts, array_cost_eval_time = run_pelt_array_based(
            cost,
            penalty=penalty,
            min_segment_length=min_segment_length,
        )
        array_pelt_time = time.perf_counter() - start_time
        array_based_overhead = array_pelt_time - array_cost_eval_time

        # Check that the implementations produce the same results
        np.testing.assert_array_equal(array_cpts, masked_cpts)
        np.testing.assert_array_almost_equal(
            pelt_costs_array_based, pelt_costs_masked, decimal=10
        )

        results.append(
            {
                "min_segment_length": min_segment_length,
                "runtime_speedup": array_pelt_time / masked_pelt_time
                if masked_pelt_time > 0
                else float("inf"),
                "overhead_speedup": array_based_overhead / masked_overhead
                if masked_overhead > 0
                else float("inf"),
                "run_pelt_time": masked_pelt_time,
                "run_pelt_array_based_time": array_pelt_time,
                "masked_cost_eval_time": masked_cost_eval_time,
                "array_cost_eval_time": array_pelt_time,
            }
        )

    # Print results in a nice table:
    df = pd.DataFrame(results)
    print("\nPELT Implementation Benchmark Results:")
    print(df.iloc[:, [0, 1, 2]])

    # Assert that array-based implementation is generally faster
    assert all(r["overhead_speedup"] > 1.0 for r in results), (
        "Array-based implementation should be faster for at least some cases"
    )


@pytest.mark.parametrize("min_segment_length", [1])
def test_old_pelt_vs_optimal_partitioning(
    cost: BaseCost, penalty: float, min_segment_length
):
    cost.fit(changepoint_data)
    pelt_costs, pelt_changepoints = run_pelt_old(
        cost=cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    # Updated definition of PELT costs, include penalty
    # for first segment.
    pelt_costs += penalty

    cost.fit(changepoint_data)
    opt_part_costs, opt_part_changepoints = run_optimal_partitioning(
        cost=cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    assert pelt_changepoints == opt_part_changepoints
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)


@pytest.mark.xfail(strict=True)
def test_xfail_old_pelt_vs_optimal_partitioning_scores(
    cost: BaseCost, penalty: float, min_segment_length=2
):
    """
    The old PELT implementation does not match the optimal partitioning
    when the segment length is greater than 1.
    """
    X = changepoint_data
    cost.fit(X)
    pelt_costs, _ = run_pelt_old(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    # Updated definition of PELT costs, include penalty
    # for first segment.
    pelt_costs += penalty

    cost.fit(X)
    opt_part_costs, _ = run_optimal_partitioning(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)


def test_old_pelt_vs_optimal_partitioning_change_points(
    cost: BaseCost, penalty: float, min_segment_length=2
):
    X = changepoint_data
    cost.fit(X)
    _, pelt_changepoints = run_pelt_old(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    cost.fit(X)
    _, opt_part_changepoints = run_optimal_partitioning(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    assert np.all(pelt_changepoints == opt_part_changepoints)


def test_run_old_pelt(cost: BaseCost, penalty: float, min_segment_length: int = 1):
    cost.fit(changepoint_data)
    pelt_costs, changepoints = run_pelt_old(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    # Assert monotonicity of costs:
    assert np.all(np.diff(pelt_costs) >= 0)
    assert len(changepoints) == n_segments - 1
    assert changepoints == [seg_len - 1]


def test_run_optimal_partitioning(
    cost: BaseCost, penalty: float, min_segment_length: int = 1
):
    cost.fit(changepoint_data)
    opt_costs, changepoints = run_optimal_partitioning(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    # Assert monotonicity of costs:
    if min_segment_length == 1:
        assert np.all(np.diff(opt_costs) >= 0)
    assert len(changepoints) == n_segments - 1
    assert changepoints == [seg_len - 1]


def test_run_pelt(cost: BaseCost, penalty: float, min_segment_length=1):
    cost.fit(changepoint_data)
    opt_costs, changepoints = run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    changepoints = changepoints - 1  # new definition in run_pelt
    assert np.all(np.diff(opt_costs) >= 0)
    assert len(changepoints) == n_segments - 1
    assert changepoints == [seg_len - 1]


def test_compare_all_pelt_functions(
    cost: BaseCost, penalty: float, min_segment_length: int = 1
):
    cost.fit(changepoint_data)
    old_pelt_costs, old_pelt_changepoints = run_pelt_old(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    # Updated definition of PELT costs, include penalty
    # for first segment.
    old_pelt_costs += penalty

    cost.fit(changepoint_data)
    opt_part_costs, opt_part_changepoints = run_optimal_partitioning(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    cost.fit(changepoint_data)
    pelt_costs, pelt_changepoints = run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    pelt_changepoints = pelt_changepoints - 1  # new definition in run_pelt

    assert old_pelt_changepoints == opt_part_changepoints == pelt_changepoints
    np.testing.assert_array_almost_equal(old_pelt_costs, opt_part_costs)
    np.testing.assert_array_almost_equal(old_pelt_costs, pelt_costs)


@pytest.mark.parametrize("min_segment_length", [1, 5, 10])
@pytest.mark.parametrize(
    "signal_end_index", list(range(20, len(alternating_sequence) + 1, 5))
)
def test_pelt_on_tricky_data(
    cost: BaseCost, penalty: float, min_segment_length: int, signal_end_index: int
):
    """
    Test PELT on a slightly more complex data set. There are
    change points every 20 samples, and the mean of the segments
    changes drastically. And the PELT implementation agrees with
    the optimal partitioning as long as the segment length is
    less than 20.
    """
    # Original "run_pelt" found 7 changepoints.
    percent_pruning_margin = 0.0
    cost.fit(alternating_sequence[0:signal_end_index])
    pelt_costs, pelt_changepoints = run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
        percent_pruning_margin=percent_pruning_margin,
    )
    opt_part_costs, opt_part_changepoints = run_optimal_partitioning(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    opt_part_changepoints = opt_part_changepoints + 1  # new definition of changepoints

    assert np.all(pelt_changepoints == opt_part_changepoints)
    np.testing.assert_almost_equal(
        pelt_costs[-1],
        pelt_partition_cost(
            alternating_sequence[0:signal_end_index],
            pelt_changepoints,
            cost,
            penalty=penalty,
        ),
        decimal=10,
        err_msg="PELT cost for final observation does not match partition cost.",
    )
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)


@pytest.mark.parametrize("min_segment_length", range(1, 20))
def test_pelt_min_segment_lengths(cost: BaseCost, penalty: float, min_segment_length):
    """
    Test PELT on a slightly more complex data set. There are
    change points every 20 samples, and the mean of the segments
    changes drastically. And the PELT implementation agrees with
    the optimal partitioning as long as the segment length is
    less than or equal to 20.

    Segment length of 30 works again...
    """
    # Original "run_pelt" found 7 changepoints.
    cost.fit(alternating_sequence)
    _, pelt_changepoints = run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    cost.fit(alternating_sequence)
    _, opt_part_changepoints = run_optimal_partitioning(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    opt_part_changepoints = opt_part_changepoints + 1  # new definition of changepoints
    assert np.all(pelt_changepoints == opt_part_changepoints)


@pytest.mark.parametrize("min_segment_length", [25] + list(range(31, 40)))
def test_pruning_margin_fixes_pelt_min_segment_length_problems(
    cost: BaseCost, penalty: float, min_segment_length
):
    """
    For all these segment lengths, the PELT implementation
    fails to find the same changepoints as the optimal partitioning.
    """
    # Original "run_pelt" found 7 changepoints.
    cost.fit(alternating_sequence)
    # The PELT implementation fails to find the same changepoints
    # as the optimal partitioning for these segment lengths,
    # when the segment length is greater than 30 and
    # the pruning margin is zero.
    pelt_costs, pelt_changepoints = run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
        percent_pruning_margin=12.0,
    )

    cost.fit(alternating_sequence)
    opt_part_costs, opt_part_changepoints = run_optimal_partitioning(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    opt_part_changepoints = opt_part_changepoints + 1  # new definition of changepoints

    assert np.all(pelt_changepoints == opt_part_changepoints)
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize("min_segment_length", [25] + list(range(31, 40)))
def test_xfail_pelt_on_tricky_data(
    cost: BaseCost, penalty: float, min_segment_length: int
):
    """
    Test PELT on a slightly more complex data set. There are
    change points every 20 samples, and the mean of the segments
    changes drastically. And the PELT implementation agrees with
    the optimal partitioning as long as the segment length is
    less than or equal to 20.

    Segment length of 30 works again...
    """
    # Original "run_pelt" found 7 changepoints.
    cost.fit(alternating_sequence)
    # The PELT implementation fails to find the same changepoints
    # as the optimal partitioning for these segment lengths,
    # when the pruning margin is zero.
    pelt_costs, _ = run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
        percent_pruning_margin=0.0,
    )

    cost.fit(alternating_sequence)
    opt_part_costs, _ = run_optimal_partitioning(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)


@pytest.mark.parametrize("min_segment_length", [1, 2, 5, 10])
def test_pelt_dense_changepoints_parametrized(cost: BaseCost, min_segment_length):
    """
    Test PELT with penalty=0.0 to ensure we get changepoints as dense as possible
    allowed by min_segment_length, for different min_segment_length values.
    """
    increasing_data = np.linspace(0, 1 * seg_len, seg_len).reshape(-1, 1)
    penalty = 0.0
    cost.fit(increasing_data)
    _, changepoints = run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    changepoints = changepoints - 1  # new definition in run_pelt
    # Expected changepoints are at every min_segment_length interval
    expected_changepoints = [
        i * min_segment_length - 1
        for i in range(1, len(increasing_data) // min_segment_length)
    ]
    assert np.all(changepoints == expected_changepoints)
