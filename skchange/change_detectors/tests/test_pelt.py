"""Tests for the Pelt implementation."""

from typing import Callable

import numpy as np
import pytest

from skchange.change_detectors.pelt import (
    get_changepoints,
    run_pelt,
)
from skchange.costs.l2_cost import L2Cost
from skchange.costs_old.cost_factory import cost_factory
from skchange.datasets.generate import generate_alternating_data
from skchange.utils.numba import njit

n_segments = 2
seg_len = 50
changepoint_data = generate_alternating_data(
    n_segments=n_segments, mean=20, segment_length=seg_len, p=1, random_state=2
)[0].values.reshape(-1, 1)

alternating_sequence = generate_alternating_data(
    n_segments=5, mean=10.5, variance=0.5, segment_length=20, p=1, random_state=5
)[0].values.reshape(-1, 1)

cost_func, cost_init_func = cost_factory("mean")
penalty = 2 * np.log(len(changepoint_data))

cost = L2Cost()


def pelt_partition_cost(
    X: np.ndarray,
    changepoints: list,
    cost_func: Callable,
    cost_init_func: Callable,
    penalty: float,
):
    cost_args = cost_init_func(X)
    n = len(X)

    total_cost = penalty * len(changepoints)
    np_changepoints = np.array(changepoints)

    interval_starts = np.concatenate((np.array([0]), np_changepoints + 1), axis=0)
    interval_ends = np.concatenate((np_changepoints, np.array([n - 1])), axis=0)

    interval_costs = cost_func(cost_args, interval_starts, interval_ends)
    total_cost += np.sum(interval_costs)

    return total_cost


@njit
def run_pelt_old(
    X: np.ndarray, cost_func, cost_init_func, penalty, min_segment_length
) -> tuple[np.ndarray, list]:
    # With 'min_segment_length' > 1, this function can return
    # segment lengths < 'min_segment_length'.
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


@njit
def run_optimal_partitioning(
    X: np.ndarray,
    cost_func,
    cost_init_func,
    penalty,
    min_segment_length: int = 1,
) -> tuple[np.ndarray, list]:
    # The simpler and more direct 'optimal partitioning' algorithm,
    # as compared to the PELT algorithm.
    params = cost_init_func(X)
    num_obs = len(X)
    min_segment_shift = min_segment_length - 1

    # Explicitly set the first element to -penalty, and the rest to NaN.
    # Last 'min_segment_shift' elements will be NaN.
    opt_cost = np.concatenate((np.array([-penalty]), np.zeros(num_obs)))
    # If min_segment_length > 1, cannot compute the cost for the first
    # [1, .., min_segment_length - 1] observations.
    # opt_cost[1:min_segment_length] = np.nan
    opt_cost[1:min_segment_length] = -penalty

    # Compute the optimal cost for the first
    # [min_segment_length, .., 2* min_segment_length - 1] observations
    # directly from the cost function, as we cannot have a changepoint
    # within the first [min_segment_length, 2*min_segment_length - 1]
    # observations.
    non_changepoint_starts = np.zeros(min_segment_length, dtype=np.int64)
    non_changepoint_ends = np.arange(min_segment_length - 1, 2 * min_segment_length - 1)

    # Shifted by 1 to account for the first element being -penalty:
    opt_cost[min_segment_length : (2 * min_segment_length)] = cost_func(
        params, non_changepoint_starts, non_changepoint_ends
    )

    # Store the previous changepoint for each last start added.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = np.repeat(-1, num_obs)

    # Evolving set of admissible segment starts.
    # Always include [0] as the start of a contiguous segment.
    candidate_starts = np.array(([0]), dtype=np.int64)

    opt_cost_observation_indices = np.arange(
        2 * min_segment_length - 1, num_obs
    ).reshape(-1, 1)

    for opt_cost_obs_index in opt_cost_observation_indices:
        segment_start = opt_cost_obs_index - min_segment_shift

        # Add the next start to the admissible starts set:
        candidate_starts = np.concatenate((candidate_starts, segment_start))
        candidate_ends = np.repeat(opt_cost_obs_index, len(candidate_starts))

        candidate_opt_costs = (
            opt_cost[candidate_starts]  # Shifted by one.
            + cost_func(params, candidate_starts, candidate_ends)
            + penalty
        )

        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        opt_cost[opt_cost_obs_index + 1] = candidate_opt_costs[argmin_candidate_cost]
        prev_cpts[opt_cost_obs_index] = candidate_starts[argmin_candidate_cost] - 1

    return opt_cost[1:], get_changepoints(prev_cpts)


@pytest.mark.parametrize("min_segment_length", [1])
def test_old_pelt_vs_optimal_partitioning(min_segment_length):
    pelt_costs, pelt_changepoints = run_pelt_old(
        changepoint_data,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    opt_part_costs, opt_part_changepoints = run_optimal_partitioning(
        changepoint_data,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    assert pelt_changepoints == opt_part_changepoints
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)


@pytest.mark.xfail
def test_xfail_old_pelt_vs_optimal_partitioning_scores(min_segment_length=2):
    """
    The old PELT implementation does not match the optimal partitioning
    when the segment length is greater than 1.
    """
    X = changepoint_data
    pelt_costs, _ = run_pelt_old(
        X,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    opt_part_costs, _ = run_optimal_partitioning(
        X,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)


def test_old_pelt_vs_optimal_partitioning_change_points(min_segment_length=2):
    X = changepoint_data
    _, pelt_changepoints = run_pelt_old(
        X,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    _, opt_part_changepoints = run_optimal_partitioning(
        X,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    assert pelt_changepoints == opt_part_changepoints


def test_run_old_pelt(min_segment_length=1):
    pelt_costs, changepoints = run_pelt_old(
        changepoint_data,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    # Assert monotonicity of costs:
    assert np.all(np.diff(pelt_costs) >= 0)
    assert len(changepoints) == n_segments - 1
    assert changepoints == [seg_len - 1]


def test_run_optimal_partitioning(min_segment_length=1):
    opt_costs, changepoints = run_optimal_partitioning(
        changepoint_data,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    # Assert monotonicity of costs:
    if min_segment_length == 1:
        assert np.all(np.diff(opt_costs) >= 0)
    assert len(changepoints) == n_segments - 1
    assert changepoints == [seg_len - 1]


def test_run_pelt(min_segment_length=1):
    opt_costs, changepoints = run_pelt(
        changepoint_data,
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    assert np.all(np.diff(opt_costs) >= 0)
    assert len(changepoints) == n_segments - 1
    assert changepoints == [seg_len - 1]


def test_compare_all_pelt_functions(min_segment_length=1):
    old_pelt_costs, old_pelt_changepoints = run_pelt_old(
        changepoint_data,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    opt_part_costs, opt_part_changepoints = run_optimal_partitioning(
        changepoint_data,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    pelt_costs, pelt_changepoints = run_pelt(
        changepoint_data,
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    assert old_pelt_changepoints == opt_part_changepoints == pelt_changepoints
    np.testing.assert_array_almost_equal(old_pelt_costs, opt_part_costs)
    np.testing.assert_array_almost_equal(old_pelt_costs, pelt_costs)


@pytest.mark.parametrize("min_segment_length", [1, 5, 10, 20])
def test_pelt_on_tricky_data(min_segment_length):
    """
    Test PELT on a slightly more complex data set. There are
    change points every 20 samples, and the mean of the segments
    changes drastically. And the PELT implementation agrees with
    the optimal partitioning as long as the segment length is
    less than or equal to 20.
    """
    # Original "run_pelt" found 7 changepoints.
    pelt_costs, pelt_changepoints = run_pelt(
        alternating_sequence,
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    opt_part_costs, opt_part_changepoints = run_optimal_partitioning(
        alternating_sequence,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    assert pelt_changepoints == opt_part_changepoints
    np.testing.assert_almost_equal(
        pelt_costs[-1],
        pelt_partition_cost(
            alternating_sequence,
            pelt_changepoints,
            cost_func,
            cost_init_func,
            penalty=penalty,
        ),
        decimal=10,
        err_msg="PELT cost for final observation does not match partition cost.",
    )
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)


@pytest.mark.parametrize("min_segment_length", range(1, 30))
def test_pelt_min_segment_lengths(min_segment_length):
    """
    Test PELT on a slightly more complex data set. There are
    change points every 20 samples, and the mean of the segments
    changes drastically. And the PELT implementation agrees with
    the optimal partitioning as long as the segment length is
    less than or equal to 20.

    Segment length of 30 works again...
    """
    # Original "run_pelt" found 7 changepoints.
    _, pelt_changepoints = run_pelt(
        alternating_sequence,
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    _, opt_part_changepoints = run_optimal_partitioning(
        alternating_sequence,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    assert pelt_changepoints == opt_part_changepoints


@pytest.mark.xfail
@pytest.mark.parametrize("min_segment_length", range(31, 40))
def test_xfail_pelt_min_segment_lengths(min_segment_length):
    """
    For all these segment lengths, the PELT implementation
    fails to find the same changepoints as the optimal partitioning.
    """
    # Original "run_pelt" found 7 changepoints.
    _, pelt_changepoints = run_pelt(
        alternating_sequence,
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    _, opt_part_changepoints = run_optimal_partitioning(
        alternating_sequence,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    assert pelt_changepoints == opt_part_changepoints


@pytest.mark.xfail
@pytest.mark.parametrize("min_segment_length", [25] + list(range(31, 40)))
def test_xfail_pelt_on_tricky_data(min_segment_length):
    """
    Test PELT on a slightly more complex data set. There are
    change points every 20 samples, and the mean of the segments
    changes drastically. And the PELT implementation agrees with
    the optimal partitioning as long as the segment length is
    less than or equal to 20.

    Segment length of 30 works again...
    """
    # Original "run_pelt" found 7 changepoints.
    pelt_costs, _ = run_pelt(
        alternating_sequence,
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    opt_part_costs, _ = run_optimal_partitioning(
        alternating_sequence,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)


@pytest.mark.parametrize("min_segment_length", [1, 2, 5, 10])
def test_pelt_dense_changepoints_parametrized(min_segment_length):
    """
    Test PELT with penalty=0.0 to ensure we get changepoints as dense as possible
    allowed by min_segment_length, for different min_segment_length values.
    """
    increasing_data = np.linspace(0, 1 * seg_len, seg_len).reshape(-1, 1)
    penalty = 0.0
    _, changepoints = run_pelt(
        increasing_data,
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    # Expected changepoints are at every min_segment_length interval
    expected_changepoints = [
        i * min_segment_length - 1
        for i in range(1, len(increasing_data) // min_segment_length)
    ]
    assert changepoints == expected_changepoints
