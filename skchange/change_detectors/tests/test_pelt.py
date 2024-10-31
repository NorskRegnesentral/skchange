"""Tests for the Pelt implementation."""

import numpy as np
import pytest
from numba import njit

from skchange.change_detectors.pelt import (
    get_changepoints,
    pelt_partition_cost,
    run_optimal_partitioning,
    run_pelt,
)
from skchange.costs.cost_factory import cost_factory
from skchange.datasets.generate import generate_alternating_data

n_segments = 2
seg_len = 50
changepoint_data = generate_alternating_data(
    n_segments=n_segments, mean=20, segment_length=seg_len, p=1, random_state=2
)[0].values.reshape(-1, 1)

alternating_sequence = generate_alternating_data(
    n_segments=5, segment_length=20, p=1, random_state=5, mean=10.5, variance=0.5
)[0].values.reshape(-1, 1)

cost_func, cost_init_func = cost_factory("mean")
penalty = 2 * np.log(len(changepoint_data))


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


@pytest.mark.parametrize("min_segment_length", [1])
def test_pelt_vs_optimal_partitioning(min_segment_length):
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
def test_xfail_pelt_vs_optimal_partitioning(min_segment_length=2):
    X = changepoint_data.values.reshape(-1, 1)
    pelt_costs, pelt_changepoints = run_pelt_old(
        X,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    opt_part_costs, opt_part_changepoints = run_optimal_partitioning(
        X,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    assert pelt_changepoints == opt_part_changepoints
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)


def test_run_pelt_old(min_segment_length=1):
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
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    assert np.all(np.diff(opt_costs) >= 0)
    assert len(changepoints) == n_segments - 1
    assert changepoints == [seg_len - 1]


def test_compare_pelt_functions(min_segment_length=1):
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
    pelt_cand_2_costs, pelt_cand_2_changepoints = run_pelt(
        changepoint_data,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    assert pelt_changepoints == opt_part_changepoints == pelt_cand_2_changepoints
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)
    np.testing.assert_array_almost_equal(pelt_costs, pelt_cand_2_costs)


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
        cost_func,
        cost_init_func,
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


@pytest.mark.xfail
def test_xfail_pelt_on_tricky_data(min_segment_length=25):
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
        cost_func,
        cost_init_func,
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
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)