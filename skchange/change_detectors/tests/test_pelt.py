"""Tests for the Pelt implementation."""

from skchange.datasets.generate import generate_alternating_data
from skchange.scores.score_factory import VALID_CHANGE_SCORES
import numpy as np
import pytest
from skchange.costs.cost_factory import cost_factory

from skchange.change_detectors.pelt import (
    run_pelt,
    run_optimal_partitioning,
    run_pelt_candidate_2,
)

n_segments = 2
seg_len = 50
changepoint_data = generate_alternating_data(
    n_segments=n_segments, mean=20, segment_length=seg_len, p=1, random_state=2
)[0]

cost_func, cost_init_func = cost_factory("mean")
penalty = 2 * np.log(len(changepoint_data))


def test_pelt_vs_optimal_partitioning(min_segment_length=1):
    X = changepoint_data.values.reshape(-1, 1)
    pelt_costs, pelt_changepoints = run_pelt(
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


def test_run_pelt(min_segment_length=1):
    X = changepoint_data.values.reshape(-1, 1)
    pelt_costs, changepoints = run_pelt(
        X,
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
    X = changepoint_data.values.reshape(-1, 1)
    opt_costs, changepoints = run_optimal_partitioning(
        X,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    # Assert monotonicity of costs:
    assert np.all(np.diff(opt_costs) >= 0)
    assert len(changepoints) == n_segments - 1
    assert changepoints == [seg_len - 1]


def test_run_pelt_candidate_2(min_segment_length=1):
    X = changepoint_data.values.reshape(-1, 1)
    opt_costs, changepoints = run_pelt_candidate_2(
        X,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    assert np.all(np.diff(opt_costs) >= 0)
    assert len(changepoints) == n_segments - 1
    assert changepoints == [seg_len - 1]


def test_compare_pelt_functions(min_segment_length=1):
    X = changepoint_data.values.reshape(-1, 1)
    pelt_costs, pelt_changepoints = run_pelt(
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
    pelt_cand_2_costs, pelt_cand_2_changepoints = run_pelt_candidate_2(
        X,
        cost_func,
        cost_init_func,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    assert pelt_changepoints == opt_part_changepoints == pelt_cand_2_changepoints
    np.testing.assert_array_almost_equal(pelt_costs, opt_part_costs)
    np.testing.assert_array_almost_equal(pelt_costs, pelt_cand_2_costs)
