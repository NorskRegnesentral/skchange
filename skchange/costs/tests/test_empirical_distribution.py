# %%
import time

import numpy as np
import pytest

from skchange.costs._empirical_distribution_cost import (
    approximate_empirical_distribution_cost,
    evaluate_empirical_distribution_function,
    make_cumulative_edf_cache,
    make_edf_cost_approximation_cache,
    pre_cached_k_term_empirical_distribution_cost_approximation,
)
from skchange.utils.numba import numba_available


# %%
def direct_empirical_distribution_cost(
    xs: np.ndarray, segment_starts: np.ndarray, segment_ends: np.ndarray
) -> np.ndarray:
    """
    Compute the empirical distribution cost for a sequence of values
    given a set of changepoints.

    Parameters
    ----------
    xs : np.ndarray
        The input data array.
    segment_starts : np.ndarray
        The start indices of the segments.
    segment_ends : np.ndarray
        The end indices of the segments.

    Returns
    -------
    np.ndarray
        1D array of empirical distribution costs for each segment defined by `cuts`.
    """
    edf_eval_points = np.sort(xs)[1:-1]  # Exclude the first and last samples
    segment_edf_per_sample = np.zeros(len(xs) - 2, dtype=np.float64)
    reciprocal_full_data_cdf_weights = len(xs) / (
        np.arange(2, len(xs), dtype=np.float64)
        * np.arange(len(xs) - 1, 1, -1, dtype=np.float64)
    )

    segment_costs = np.zeros(len(segment_starts), dtype=np.float64)
    for i, (segment_start, segment_end) in enumerate(zip(segment_starts, segment_ends)):
        segment_length = segment_end - segment_start
        segment_data = xs[segment_start:segment_end]

        segment_edf_per_sample[:] = evaluate_empirical_distribution_function(
            segment_data,
            edf_eval_points,
        )

        # Apply continuity correction:
        segment_edf_per_sample -= 1 / (2 * segment_length)

        # Clip to avoid log(0) issues:
        segment_edf_per_sample = np.clip(segment_edf_per_sample, 1e-10, 1 - 1e-10)
        one_minus_segment_empirical_distribution_per_sample = (
            1.0 - segment_edf_per_sample
        )

        integrated_ll_at_mle = segment_length * (
            np.sum(
                (
                    segment_edf_per_sample * np.log(segment_edf_per_sample)
                    + one_minus_segment_empirical_distribution_per_sample
                    * np.log(one_minus_segment_empirical_distribution_per_sample)
                )
                * reciprocal_full_data_cdf_weights
            )
        )

        # The cost is the negative integrated log-likelihood:
        segment_costs[i] = -integrated_ll_at_mle

    return segment_costs


def test_evaluate_empirical_distribution_function():
    xs = np.array([1, 2, 3, 4, 5])

    edf_eval_points_1 = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    xs_edf_eval_point_quantiles = evaluate_empirical_distribution_function(
        xs, edf_eval_points_1
    )
    np.testing.assert_allclose(
        xs_edf_eval_point_quantiles,
        np.array([0.2, 0.4, 0.4, 0.6, 0.6]),
    )

    edf_eval_points_2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    xs_edf_eval_point_quantiles_equal = evaluate_empirical_distribution_function(
        xs, edf_eval_points_2
    )
    np.testing.assert_allclose(
        xs_edf_eval_point_quantiles_equal,
        np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
    )

    # Add more tests with different data
    xs2 = np.array([1, 1, 2, 2, 3, 3])
    edf_eval_points2 = np.array([0.99, 1.99, 2.99])
    xs2_edf_eval_point_quantiles = evaluate_empirical_distribution_function(
        xs2, edf_eval_points2
    )
    np.testing.assert_allclose(
        xs2_edf_eval_point_quantiles,
        np.array([0.0, 1 / 3, 2 / 3]),
    )


def test_evaluate_edf_from_cache():
    xs = np.array([1, 2, 3, 4, 5])

    edf_eval_points_1 = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    approx_cost_cache = make_cumulative_edf_cache(xs, edf_eval_points_1)
    xs_edf_eval_point_quantiles = (
        approx_cost_cache[:, -1] - approx_cost_cache[:, 0]
    ) / len(xs)
    np.testing.assert_array_equal(
        xs_edf_eval_point_quantiles,
        np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
    )

    edf_eval_points_2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    approx_cost_cache_2 = make_cumulative_edf_cache(xs, edf_eval_points_2)
    xs_edf_eval_point_quantiles_2 = (
        approx_cost_cache_2[:, -1] - approx_cost_cache_2[:, 0]
    ) / len(xs)
    np.testing.assert_array_equal(
        xs_edf_eval_point_quantiles_2,
        np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
    )


def test_k_term_empirical_distribution_cost_approximation():
    xs = np.array([1, 2, 3, 4, 5])
    segment_starts = np.array([0, 2])
    segment_ends = np.array([2, 5])
    num_approx_quantiles = 3
    approx_cost = approximate_empirical_distribution_cost(
        xs, segment_starts, segment_ends, num_approx_quantiles
    )
    direct_cost = direct_empirical_distribution_cost(xs, segment_starts, segment_ends)
    approx_cost_cache = make_edf_cost_approximation_cache(xs, num_approx_quantiles)
    pre_cached_approx_cost = (
        pre_cached_k_term_empirical_distribution_cost_approximation(
            approx_cost_cache, segment_starts, segment_ends
        )
    )
    print(f"Direct cost: {direct_cost}")
    print(f"Approximate cost: {approx_cost}")
    print(f"Pre-cached approximate cost: {pre_cached_approx_cost}")

    # Add more tests with different data and changepoints
    xs2 = np.array([1, 1, 2, 2, 3, 3])
    segment_starts2 = np.array([0, 3])
    segment_ends2 = np.array([3, 6])
    k2 = 2
    approx_cost2 = approximate_empirical_distribution_cost(
        xs2, segment_starts2, segment_ends2, k2
    )
    approx_cost_cache_2 = make_edf_cost_approximation_cache(xs2, k2)
    pre_cached_approx_cost2 = (
        pre_cached_k_term_empirical_distribution_cost_approximation(
            approx_cost_cache_2, segment_starts2, segment_ends2
        )
    )
    direct_cost2 = direct_empirical_distribution_cost(
        xs2, segment_starts2, segment_ends2
    )

    print(f"Direct cost for xs2: {direct_cost2}")
    print(f"Approximate cost for xs2: {approx_cost2}")
    print(f"Pre-cached approximate cost for xs2: {pre_cached_approx_cost2}")

    no_change_cuts = np.array([[0, len(xs)]])
    no_change_starts = no_change_cuts[:, 0]
    no_change_ends = no_change_cuts[:, 1]
    no_change_approx_cost = approximate_empirical_distribution_cost(
        xs, no_change_starts, no_change_ends, num_approx_quantiles
    )
    no_change_approx_cost_cache = make_edf_cost_approximation_cache(
        xs, num_approx_quantiles
    )
    no_change_approx_cost_pre_cached = (
        pre_cached_k_term_empirical_distribution_cost_approximation(
            no_change_approx_cost_cache, no_change_starts, no_change_ends
        )
    )
    no_change_direct_cost = direct_empirical_distribution_cost(
        xs, no_change_starts, no_change_ends
    )
    print(f"No change direct cost: {no_change_direct_cost}")
    print(f"No change approximate cost: {no_change_approx_cost}")
    print(f"No change pre-cached approximate cost: {no_change_approx_cost_pre_cached}")


@pytest.mark.parametrize(
    ["tolerance", "n_samples"], [(0.15, 60), (0.10, 120), (0.05, 1000)]
)
def test_approximate_vs_direct_cost_on_longer_data(tolerance: float, n_samples: int):
    np.random.seed(42)  # For reproducibility
    first_segment = np.random.normal(size=n_samples)
    second_segment = np.random.normal(
        size=n_samples, loc=5
    )  # Shifted mean for the second segment
    xs = np.concatenate([first_segment, second_segment])

    correct_segment_starts = np.array([0, n_samples])
    correct_segment_ends = np.array([n_samples, len(xs)])
    no_change_segment_starts = np.array([0])
    no_change_segment_ends = np.array([len(xs)])
    # no_change_cuts = np.array([[0, len(xs)]])

    # Suggested value based on the length of xs:
    num_quantiles = int(4 * np.log(len(xs)))

    one_change_approx_costs = approximate_empirical_distribution_cost(
        xs, correct_segment_starts, correct_segment_ends, num_quantiles=num_quantiles
    )
    one_change_direct_costs = direct_empirical_distribution_cost(
        xs, correct_segment_starts, correct_segment_ends
    )

    relative_differences = np.abs(
        (one_change_approx_costs - one_change_direct_costs) / one_change_direct_costs
    )
    assert np.all(relative_differences < tolerance), (
        f"Relative differences exceed {tolerance * 100}%: {relative_differences}"
    )
    # print(f"Direct cost on longer data: {one_change_direct_costs}")
    # print(f"Approximate cost on longer data: {one_change_approx_costs}")

    single_segment_approx_cost = approximate_empirical_distribution_cost(
        xs,
        no_change_segment_starts,
        no_change_segment_ends,
        num_quantiles=num_quantiles,
    )
    single_segment_direct_cost = direct_empirical_distribution_cost(
        xs, no_change_segment_starts, no_change_segment_ends
    )
    single_segment_relative_difference = np.abs(
        (single_segment_approx_cost[0] - single_segment_direct_cost[0])
        / single_segment_direct_cost[0]
    )
    assert single_segment_relative_difference < tolerance, (
        f"Relative difference for single segment exceeds {tolerance * 100}%: "
        f"{single_segment_relative_difference}"
    )

    assert single_segment_approx_cost - np.sum(one_change_approx_costs) > 0, (
        "Approximate cost for no change should be greater than for two segments."
    )
    assert single_segment_direct_cost - np.sum(one_change_direct_costs) > 0, (
        "Direct cost for no change should be greater than for two segments."
    )
    # print(f"Direct cost for single segment: {single_segment_direct_cost}")
    # print(f"Approximate cost for single segment: {single_segment_approx_cost}")


@pytest.mark.parametrize(
    ["rel_tol", "n_samples"], [(5.0e-2, 60), (1.0e-3, 120), (1.0e-4, 1000)]
)
def test_approximate_vs_precached_approximate_cost(rel_tol: float, n_samples: int):
    np.random.seed(42)  # For reproducibility
    first_segment = np.random.normal(size=n_samples)
    second_segment = np.random.normal(size=n_samples, loc=5)
    xs = np.concatenate([first_segment, second_segment])
    num_quantiles = int(4 * np.log(len(xs)))

    correct_cuts = np.array([[0, n_samples], [n_samples, len(xs)]])
    correct_segment_starts = correct_cuts[:, 0]
    correct_segment_ends = correct_cuts[:, 1]

    no_change_cuts = np.array([[0, len(xs)]])
    no_change_segment_starts = no_change_cuts[:, 0]
    no_change_segment_ends = no_change_cuts[:, 1]

    # Compare approximate vs precached on correct cuts
    approx_costs = approximate_empirical_distribution_cost(
        xs, correct_segment_starts, correct_segment_ends, num_quantiles=num_quantiles
    )
    approx_cost_cache = make_edf_cost_approximation_cache(xs, num_quantiles)
    precached_costs = pre_cached_k_term_empirical_distribution_cost_approximation(
        approx_cost_cache, correct_segment_starts, correct_segment_ends
    )
    np.testing.assert_allclose(approx_costs, precached_costs, rtol=rel_tol)

    # Compare on single segment
    single_approx_cost = approximate_empirical_distribution_cost(
        xs,
        no_change_segment_starts,
        no_change_segment_ends,
        num_quantiles=num_quantiles,
    )
    single_precached_cost = pre_cached_k_term_empirical_distribution_cost_approximation(
        approx_cost_cache, no_change_segment_starts, no_change_segment_ends
    )
    np.testing.assert_allclose(single_approx_cost, single_precached_cost, rtol=rel_tol)


# %%
def test_direct_vs_approximation_runtime(n_samples=10_000):
    xs = np.random.normal(size=n_samples)
    per_hundred_step_cuts = np.array(
        [[i * 100, (i + 1) * 100] for i in range(len(xs) // 100)]
    )
    per_hundred_step_segment_starts = per_hundred_step_cuts[:, 0]
    per_hundred_step_segment_ends = per_hundred_step_cuts[:, 1]
    num_approx_quantiles = int(4 * np.log(n_samples))

    # Call once in case of JIT compilation overhead:
    direct_cost = direct_empirical_distribution_cost(
        xs, per_hundred_step_segment_starts, per_hundred_step_segment_ends
    )
    start_time = time.perf_counter()
    direct_cost = direct_empirical_distribution_cost(
        xs, per_hundred_step_segment_starts, per_hundred_step_segment_ends
    )
    end_time = time.perf_counter()
    direct_cost_eval_time = end_time - start_time
    total_direct_cost = np.sum(direct_cost)

    # print(
    #     f"Total direct cost: {total_direct_cost}, Time taken: {direct_cost_eval_time:.4e} seconds"
    # )
    assert direct_cost_eval_time < 5.0e-2, (
        "Direct evaluation time should be less than 0.05 seconds."
    )

    ### Approximate evaluation:
    # Call once in case of JIT compilation overhead:
    approximate_empirical_distribution_cost(
        xs,
        per_hundred_step_segment_starts,
        per_hundred_step_segment_ends,
        num_quantiles=num_approx_quantiles,
    )
    start_time = time.perf_counter()
    approx_cost = approximate_empirical_distribution_cost(
        xs,
        per_hundred_step_segment_starts,
        per_hundred_step_segment_ends,
        num_quantiles=num_approx_quantiles,
    )
    end_time = time.perf_counter()
    approximate_cost_eval_time = end_time - start_time
    total_approx_cost = np.sum(approx_cost)

    # print(
    #     f"Total approximate cost: {total_approx_cost}, "
    #     f"Time taken: {approximate_cost_eval_time:.4e} sec."
    # )
    assert approximate_cost_eval_time < 1.0e-2, (
        "Approximate evaluation time should be less than 0.01 sec."
    )

    ### Pre-caching the approximation:
    # Call once in case of JIT compilation overhead:
    approx_cost_cache = make_edf_cost_approximation_cache(xs, num_approx_quantiles)
    pre_cached_k_term_empirical_distribution_cost_approximation(
        approx_cost_cache,
        per_hundred_step_segment_starts,
        per_hundred_step_segment_ends,
    )

    cache_start_time = time.perf_counter()
    approx_cost_cache = make_edf_cost_approximation_cache(xs, num_approx_quantiles)
    cache_end_time = time.perf_counter()
    cache_creation_time = cache_end_time - cache_start_time

    start_time = time.perf_counter()
    pre_cached_cost = pre_cached_k_term_empirical_distribution_cost_approximation(
        approx_cost_cache,
        per_hundred_step_segment_starts,
        per_hundred_step_segment_ends,
    )
    end_time = time.perf_counter()
    pre_cached_eval_time = end_time - start_time
    total_pre_cached_cost = np.sum(pre_cached_cost)

    # print(f"Cache creation time: {cache_creation_time:.4e} seconds")
    # print(
    #     f"Total pre-cached approximate cost: {total_pre_cached_cost}, "
    #     f"Time taken: {pre_cached_eval_time:.4e} seconds"
    # )
    if numba_available:
        max_cache_creation_time = 1.0e-2
        max_pre_cached_eval_time = 5.0e-4
    else:
        max_cache_creation_time = 5.0e-1
        max_pre_cached_eval_time = 5.0e-2

    assert cache_creation_time < max_cache_creation_time, (
        f"Cache creation should take less than {max_cache_creation_time:.2e} seconds."
    )
    assert pre_cached_eval_time < max_pre_cached_eval_time, (
        f"Pre-cached eval. should take less than {max_pre_cached_eval_time:.2e} sec."
    )

    assert np.isclose(total_pre_cached_cost, total_approx_cost, rtol=1.0e-4), (
        "Pre-cached approximate cost does not match approximate cost within tolerance."
    )
    assert np.isclose(total_direct_cost, total_pre_cached_cost, rtol=5.0e-2), (
        "Approximate cost does not match direct cost within tolerance."
    )
