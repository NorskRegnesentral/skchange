"""Tests for the PELT implementation."""

import numpy as np
import pytest

from skchange.change_detectors.pelt import (
    get_changepoints,
    run_pelt,
)
from skchange.costs import BaseCost, L2Cost
from skchange.datasets.generate import generate_alternating_data

n_segments = 2
seg_len = 50
changepoint_data = generate_alternating_data(
    n_segments=n_segments, mean=20, segment_length=seg_len, p=1, random_state=2
).values.reshape(-1, 1)

alternating_sequence = generate_alternating_data(
    n_segments=5, mean=10.5, variance=0.5, segment_length=20, p=1, random_state=5
).values.reshape(-1, 1)


@pytest.fixture
def testing_cost():
    return L2Cost()


@pytest.fixture
def testing_penalty():
    return 2 * np.log(len(changepoint_data))


def pelt_partition_cost(
    X: np.ndarray,
    changepoints: np.ndarray,
    cost: BaseCost,
    penalty: float,
):
    cost.fit(X)
    n = len(X)

    total_cost = penalty * len(changepoints)
    np_changepoints = np.asarray(changepoints)

    interval_starts = np.concatenate((np.array([0]), np_changepoints + 1), axis=0)
    interval_ends = np.concatenate((np_changepoints, np.array([n - 1])), axis=0)

    interval_costs = np.sum(
        cost.evaluate(np.column_stack((interval_starts, interval_ends + 1))), axis=1
    )
    total_cost += np.sum(interval_costs)

    return total_cost


def run_pelt_old(
    num_samples: int, cost: BaseCost, penalty, min_segment_length
) -> tuple[np.ndarray, list]:
    # With 'min_segment_length' > 1, this function can return
    # segment lengths < 'min_segment_length'.
    assert cost.is_fitted, "Cost function must be fitted before running PELT."

    starts = np.array((), dtype=np.int64)  # Evolving set of admissible segment starts.
    init_starts = np.zeros(min_segment_length - 1, dtype=np.int64)
    init_ends = np.arange(min_segment_length - 1)
    opt_cost = np.zeros(num_samples + 1) - penalty
    opt_cost[1:min_segment_length] = np.sum(
        cost.evaluate(np.column_stack((init_starts, init_ends + 1))), axis=1
    )

    # Store the previous changepoint for each t.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = np.repeat(-1, num_samples)

    ts = np.arange(min_segment_length - 1, num_samples).reshape(-1, 1)
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
    num_obs: int,
    cost: BaseCost,
    penalty,
    min_segment_length: int = 1,
) -> tuple[np.ndarray, list]:
    # The simpler and more direct 'optimal partitioning' algorithm,
    # as compared to the PELT algorithm.
    assert cost.is_fitted, "Cost function must be fitted before running Opt. Part."
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
    opt_cost[min_segment_length : (2 * min_segment_length)] = np.sum(
        cost.evaluate(
            np.column_stack((non_changepoint_starts, non_changepoint_ends + 1))
        ),
        axis=1,
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


@pytest.mark.parametrize("min_segment_length", [1])
def test_old_pelt_vs_optimal_partitioning(
    testing_cost: BaseCost, testing_penalty, min_segment_length
):
    testing_cost.fit(changepoint_data)
    testing_cost.adapt(changepoint_data)

    pelt_costs, pelt_changepoints = run_pelt_old(
        len(changepoint_data),
        cost=testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )

    opt_part_costs, opt_part_changepoints = run_optimal_partitioning(
        len(changepoint_data),
        cost=testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )

    assert pelt_changepoints == opt_part_changepoints
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)


@pytest.mark.xfail
def test_xfail_old_pelt_vs_optimal_partitioning_scores(
    testing_cost: BaseCost, testing_penalty, min_segment_length=2
):
    """
    The old PELT implementation does not match the optimal partitioning
    when the segment length is greater than 1.
    """
    num_samples = len(changepoint_data)
    testing_cost.fit(num_samples)
    testing_cost.fit(num_samples)

    pelt_costs, _ = run_pelt_old(
        num_samples,
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    opt_part_costs, _ = run_optimal_partitioning(
        num_samples,
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)


def test_old_pelt_vs_optimal_partitioning_change_points(
    testing_cost: BaseCost, testing_penalty, min_segment_length=2
):
    testing_cost.fit(changepoint_data)
    testing_cost.adapt(changepoint_data)

    num_samples = len(changepoint_data)
    _, pelt_changepoints = run_pelt_old(
        num_samples,
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    _, opt_part_changepoints = run_optimal_partitioning(
        num_samples,
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    assert pelt_changepoints == opt_part_changepoints


def test_run_old_pelt(testing_cost: BaseCost, testing_penalty, min_segment_length=1):
    testing_cost.fit(changepoint_data)
    testing_cost.adapt(changepoint_data)

    pelt_costs, changepoints = run_pelt_old(
        len(changepoint_data),
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    # Assert monotonicity of costs:
    assert np.all(np.diff(pelt_costs) >= 0)
    assert len(changepoints) == n_segments - 1
    assert changepoints == [seg_len - 1]


def test_run_optimal_partitioning(
    testing_cost: BaseCost, testing_penalty, min_segment_length=1
):
    testing_cost.fit(changepoint_data)
    testing_cost.adapt(changepoint_data)

    opt_costs, changepoints = run_optimal_partitioning(
        len(changepoint_data),
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    # Assert monotonicity of costs:
    if min_segment_length == 1:
        assert np.all(np.diff(opt_costs) >= 0)
    assert len(changepoints) == n_segments - 1
    assert changepoints == [seg_len - 1]


def test_run_pelt(testing_cost: BaseCost, testing_penalty, min_segment_length=1):
    testing_cost.fit(changepoint_data)
    testing_cost.adapt(changepoint_data)

    opt_costs, changepoints = run_pelt(
        len(changepoint_data),
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    changepoints = changepoints - 1  # new definition in run_pelt
    assert np.all(np.diff(opt_costs) >= 0)
    assert len(changepoints) == n_segments - 1
    assert changepoints == [seg_len - 1]


def test_compare_all_pelt_functions(
    testing_cost: BaseCost, testing_penalty, min_segment_length=1
):
    testing_cost.fit(changepoint_data)
    testing_cost.adapt(changepoint_data)

    num_samples = len(changepoint_data)
    old_pelt_costs, old_pelt_changepoints = run_pelt_old(
        num_samples,
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    opt_part_costs, opt_part_changepoints = run_optimal_partitioning(
        num_samples,
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    pelt_costs, pelt_changepoints = run_pelt(
        num_samples,
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    pelt_changepoints = pelt_changepoints - 1  # new definition in run_pelt

    assert old_pelt_changepoints == opt_part_changepoints == pelt_changepoints
    np.testing.assert_array_almost_equal(old_pelt_costs, opt_part_costs)
    np.testing.assert_array_almost_equal(old_pelt_costs, pelt_costs)


@pytest.mark.parametrize("min_segment_length", [1, 5, 10])
def test_pelt_on_tricky_data(
    testing_cost: BaseCost, testing_penalty, min_segment_length
):
    """
    Test PELT on a slightly more complex data set. There are
    change points every 20 samples, and the mean of the segments
    changes drastically. And the PELT implementation agrees with
    the optimal partitioning as long as the segment length is
    less than 20.
    """
    testing_cost.fit(alternating_sequence)
    testing_cost.adapt(alternating_sequence)

    # Original "run_pelt" found 7 changepoints.
    pelt_costs, pelt_changepoints = run_pelt(
        len(alternating_sequence),
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    pelt_changepoints = pelt_changepoints - 1  # new definition in run_pelt
    opt_part_costs, opt_part_changepoints = run_optimal_partitioning(
        len(alternating_sequence),
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )

    assert np.all(pelt_changepoints == opt_part_changepoints)
    np.testing.assert_almost_equal(
        pelt_costs[-1],
        pelt_partition_cost(
            alternating_sequence,
            pelt_changepoints,
            testing_cost,
            penalty=testing_penalty,
        ),
        decimal=10,
        err_msg="PELT cost for final observation does not match partition cost.",
    )
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)


@pytest.mark.parametrize("min_segment_length", range(1, 20))
def test_pelt_min_segment_lengths(
    testing_cost: BaseCost, testing_penalty, min_segment_length
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
    testing_cost.fit(alternating_sequence)
    testing_cost.adapt(alternating_sequence)

    _, pelt_changepoints = run_pelt(
        len(alternating_sequence),
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    pelt_changepoints = pelt_changepoints - 1  # new definition in run_pelt
    _, opt_part_changepoints = run_optimal_partitioning(
        len(alternating_sequence),
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    assert np.all(pelt_changepoints == opt_part_changepoints)


@pytest.mark.xfail
@pytest.mark.parametrize("min_segment_length", range(31, 40))
def test_xfail_pelt_min_segment_lengths(
    testing_cost: BaseCost, testing_penalty, min_segment_length
):
    """
    For all these segment lengths, the PELT implementation
    fails to find the same changepoints as the optimal partitioning.
    """
    # Original "run_pelt" found 7 changepoints.
    testing_cost.fit(alternating_sequence)
    testing_cost.adapt(alternating_sequence)

    _, pelt_changepoints = run_pelt(
        len(alternating_sequence),
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    pelt_changepoints = pelt_changepoints - 1  # new definition in run_pelt
    _, opt_part_changepoints = run_optimal_partitioning(
        len(alternating_sequence),
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )

    assert pelt_changepoints == opt_part_changepoints


@pytest.mark.xfail
@pytest.mark.parametrize("min_segment_length", [25] + list(range(31, 40)))
def test_xfail_pelt_on_tricky_data(
    testing_cost: BaseCost, testing_penalty, min_segment_length
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
    testing_cost.fit(alternating_sequence)
    testing_cost.adapt(alternating_sequence)
    pelt_costs, _ = run_pelt(
        len(alternating_sequence),
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    opt_part_costs, _ = run_optimal_partitioning(
        len(alternating_sequence),
        testing_cost,
        penalty=testing_penalty,
        min_segment_length=min_segment_length,
    )
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)


@pytest.mark.parametrize("min_segment_length", [1, 2, 5, 10])
def test_pelt_dense_changepoints_parametrized(
    testing_cost: BaseCost, testing_penalty, min_segment_length
):
    """
    Test PELT with penalty=0.0 to ensure we get changepoints as dense as possible
    allowed by min_segment_length, for different min_segment_length values.
    """
    increasing_data = np.linspace(0, 1 * seg_len, seg_len).reshape(-1, 1)
    penalty = 0.0
    testing_cost.fit(increasing_data)
    testing_cost.adapt(increasing_data)
    _, changepoints = run_pelt(
        len(increasing_data),
        testing_cost,
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
