# %%
import time

import numpy as np
import pytest
from scipy import stats

from skchange.change_detectors import PELT
from skchange.costs._empirical_distribution_cost import (
    EmpiricalDistributionCost,
    approximate_mle_edf_cost,
    compute_finite_difference_derivatives,
    evaluate_empirical_distribution_function,
    make_approximate_mle_edf_cost_cache,
    make_cumulative_edf_cache,
    make_fixed_cdf_cost_integration_weights,
    pre_cached_approximate_mle_edf_cost,
    pre_cached_fixed_cdf_empirical_distribution_cost,
)
from skchange.utils.numba import numba_available


# %%
# Generate 10_000 samples from two different normal distributions, concatenate them,
# and use them as input data for testing the empirical distribution cost within PELT.
def generate_test_data(n_samples: int = 1_000) -> np.ndarray:
    """Generate test data for empirical distribution cost evaluation."""
    np.random.seed(42)  # For reproducibility
    first_segment = np.random.normal(size=n_samples)
    second_segment = np.random.normal(size=n_samples, loc=5)  # Shifted mean
    return np.concatenate([first_segment, second_segment])


# test_data = generate_test_data(2_000)
# cost = EmpiricalDistributionCost(num_approximation_quantiles=10, use_cache=True)
# change_detector = PELT(cost=cost, min_segment_length=15)
# change_detector.fit(test_data)
# results = change_detector.predict(test_data)

# %%
# %%timeit
# change_detector.predict(test_data)


# %%
def direct_mle_edf_cost(
    xs: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    apply_continuity_correction: bool = False,
) -> np.ndarray:
    """Compute exact empirical distribution cost.

    This function computes the empirical distribution cost for a sequence `xs` with
    given segment cuts defined by `segment_starts` and `segment_ends`. The cost is
    computed as the integrated log-likelihood of the empirical distribution function
    (EDF) for each segment. The EDF is evaluated at the sorted values of `xs`, excluding
    the first and last samples to avoid boundary effects.

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
    assert xs.ndim == 1, "Input data must be a 1D array."
    edf_eval_points = np.sort(xs)[1:-1]  # Exclude the first and last samples
    n_samples = len(xs)
    reciprocal_full_data_cdf_weights = np.square(n_samples) / (
        np.arange(2, n_samples, dtype=np.float64)
        * np.arange(n_samples - 2, 0, -1, dtype=np.float64)
    )

    segment_costs = np.zeros(len(segment_starts), dtype=np.float64)
    for i, (segment_start, segment_end) in enumerate(zip(segment_starts, segment_ends)):
        segment_length = segment_end - segment_start
        segment_data = xs[segment_start:segment_end]

        segment_edf = evaluate_empirical_distribution_function(
            segment_data,
            edf_eval_points,
        )

        if apply_continuity_correction:
            segment_edf -= 1 / (2 * segment_length)

        # Clip to avoid log(0) issues:
        segment_edf = np.clip(segment_edf, 1e-10, 1 - 1e-10)
        one_minus_segment_edf = 1.0 - segment_edf

        integrated_ll_at_mle = (segment_length / n_samples) * (
            np.sum(
                (
                    segment_edf * np.log(segment_edf)
                    + one_minus_segment_edf * np.log(one_minus_segment_edf)
                )
                * reciprocal_full_data_cdf_weights
            )
        )

        # The cost equals twice the negative integrated log-likelihood:
        segment_costs[i] = -2.0 * integrated_ll_at_mle

    return segment_costs


def fixed_cdf_empirical_distribution_cost(
    xs: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    fixed_quantiles: np.ndarray,
    fixed_ts: np.ndarray,
    apply_continuity_correction: bool = False,
) -> np.ndarray:
    """Compute the empirical distribution cost on a refernce cdf.

    This function computes the empirical distribution cost for a sequence `xs` with
    given segment cuts defined by `segment_starts` and `segment_ends`. The cost is
    computed as the integrated log-likelihood of the empirical distribution function
    (EDF) for each segment, approximated by a sum over the provided quantiles.

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
        1D array of empirical distribution cost evaluated for fixed cdf.
    """
    assert xs.ndim == 1, "Input data must be a 1D array."

    # Compute the integrals weights by approximating the derivative of the fixed CDF:
    if len(fixed_quantiles) < 3:
        raise ValueError("At least three fixed quantile values are required.")

    one_minus_fixed_quantiles = 1.0 - fixed_quantiles
    reciprocal_fixed_cdf_weights = 1.0 / (fixed_quantiles * one_minus_fixed_quantiles)

    # Second order finite difference approximation for middle quantiles:
    # (Second order also for non-uniform quantiles)
    fixed_quantile_derivatives = compute_finite_difference_derivatives(
        ts=fixed_ts,
        ys=fixed_quantiles,
    )

    segment_costs = np.zeros(len(segment_starts), dtype=np.float64)
    for i, (segment_start, segment_end) in enumerate(zip(segment_starts, segment_ends)):
        segment_length = segment_end - segment_start
        segment_data = xs[segment_start:segment_end]

        segment_edf_per_sample = evaluate_empirical_distribution_function(
            segment_data,
            fixed_ts,
        )

        if apply_continuity_correction:
            segment_edf_per_sample -= 1 / (2 * segment_length)

        # Clip to avoid log(0) issues:
        segment_edf_per_sample = np.clip(segment_edf_per_sample, 1e-10, 1 - 1e-10)
        one_minus_segment_empirical_distribution_per_sample = (
            1.0 - segment_edf_per_sample
        )

        integrated_ll_at_mle = segment_length * (
            np.sum(
                (
                    segment_edf_per_sample * np.log(fixed_quantiles)
                    + one_minus_segment_empirical_distribution_per_sample
                    * np.log(one_minus_fixed_quantiles)
                )
                # Can pre pre-computed:
                * reciprocal_fixed_cdf_weights
                * fixed_quantile_derivatives
            )
        )

        # The cost equals twice the negative integrated log-likelihood:
        segment_costs[i] = -2.0 * integrated_ll_at_mle

    return segment_costs


def test_fixed_cdf_empirical_distribution_cost_vs_direct_cost():
    xs = np.array([1, 5, 2, 3, 4])
    # xs = np.arange(100)
    segment_starts = np.array([0, 3])
    segment_ends = np.array([3, 5])
    segment_starts = np.array([0])
    segment_ends = np.array([len(xs)])

    # Direct cost:
    direct_cost = direct_mle_edf_cost(xs, segment_starts, segment_ends)

    # Fixed CDF cost:
    quantile_ts = np.sort(xs)[1:-1]  # Exclude first and last samples
    # quantiles = np.searchsorted(xs, quantile_values, side="right") / len(xs)
    quantiles = np.arange(2, len(xs)) / len(xs)
    fixed_cdf_cost = fixed_cdf_empirical_distribution_cost(
        xs,
        segment_starts,
        segment_ends,
        fixed_quantiles=quantiles,
        fixed_ts=quantile_ts,
    )

    one_minus_fixed_quantiles = 1 - quantiles
    fixed_cdf_quantile_weights = make_fixed_cdf_cost_integration_weights(
        fixed_quantiles=quantiles,
        one_minus_fixed_quantiles=one_minus_fixed_quantiles,
        fixed_ts=quantile_ts,
    )

    fixed_cdf_cost_cached = pre_cached_fixed_cdf_empirical_distribution_cost(
        xs,
        segment_starts,
        segment_ends,
        fixed_quantiles=quantiles,
        one_minus_fixed_quantiles=one_minus_fixed_quantiles,
        fixed_ts=quantile_ts,
        quantile_integration_weights=fixed_cdf_quantile_weights,
    )

    print(f"Direct cost: {direct_cost}")
    print(f"Fixed CDF cost: {fixed_cdf_cost}")
    print(f"Fixed CDF cost cached: {fixed_cdf_cost_cached}")

    np.testing.assert_equal(direct_cost, fixed_cdf_cost)
    np.testing.assert_equal(direct_cost, fixed_cdf_cost_cached)


def test_fixed_cdf_empirical_distribution_cost():
    # Test the standard normal CDF as fixed quantiles:
    quantiles = np.array([0.025, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 0.975])
    gaussian_quantile_ts = stats.norm.ppf(
        quantiles
    )  # Inverse CDF of the standard normal distribution

    gaussian_quantile_derivatives = stats.norm.pdf(gaussian_quantile_ts)
    approx_gaussian_quantile_derivatives = compute_finite_difference_derivatives(
        ts=gaussian_quantile_ts, ys=quantiles
    )
    (
        np.testing.assert_allclose(
            gaussian_quantile_derivatives,
            approx_gaussian_quantile_derivatives,
            rtol=0.15,
            atol=1.0e-3,
        ),
        "Quantile derivatives do not match computed second-order derivatives.",
    )

    denser_quantiles = np.exp(np.linspace(-3.0, 3.0, 100)) / (
        1.0 + np.exp(np.linspace(-3.0, 3.0, 100))
    )
    dense_quantile_ts = stats.norm.ppf(denser_quantiles)
    dense_quantile_derivatives = stats.norm.pdf(dense_quantile_ts)
    approx_dense_quantile_derivatives = compute_finite_difference_derivatives(
        ts=dense_quantile_ts, ys=denser_quantiles
    )

    # With denser sampling, much higher accuracy is expected:
    np.testing.assert_allclose(
        dense_quantile_derivatives,
        approx_dense_quantile_derivatives,
        rtol=1.0e-10,
        atol=1.0e-4,
    )

    xs = stats.norm.rvs(size=500, random_state=42)
    cdf_eval_ts = np.linspace(np.min(xs), np.max(xs), 100)

    segment_starts = np.array([0])
    segment_ends = np.array([500])

    empirical_quantiles = np.clip(
        evaluate_empirical_distribution_function(xs, cdf_eval_ts), 1e-10, 1 - 1e-10
    )
    gaussian_quantiles = stats.norm.cdf(cdf_eval_ts)
    laplace_quantiles = stats.laplace.cdf(cdf_eval_ts)

    fixed_empirical_cdf_cost = fixed_cdf_empirical_distribution_cost(
        xs,
        segment_starts,
        segment_ends,
        fixed_quantiles=empirical_quantiles,
        fixed_ts=cdf_eval_ts,
    )

    fixed_gaussian_cdf_cost = fixed_cdf_empirical_distribution_cost(
        xs,
        segment_starts,
        segment_ends,
        fixed_quantiles=gaussian_quantiles,
        fixed_ts=cdf_eval_ts,
    )

    # laplacian_quantile_ts = stats.laplace.ppf(quantiles)
    fixed_laplacian_cdf_cost = fixed_cdf_empirical_distribution_cost(
        xs,
        segment_starts,
        segment_ends,
        fixed_quantiles=laplace_quantiles,
        fixed_ts=cdf_eval_ts,
    )

    # NOTE: With the inverse-cdf weighting, the fixed laplacian CDF cost is lower.
    #       Without the weighting, we get as expected:
    #       empirical_cdf_cost < gaussian_cdf_cost < laplacian_cdf_cost.
    print(f"Fixed empirical CDF cost: {fixed_empirical_cdf_cost}")
    print(f"Fixed Gaussian CDF cost: {fixed_gaussian_cdf_cost}")
    print(f"Fixed Laplacian CDF cost: {fixed_laplacian_cdf_cost}")
    assert np.all(fixed_laplacian_cdf_cost < fixed_gaussian_cdf_cost), (
        "Laplacian CDF cost should be lower than Gaussian CDF cost,"
        " on standard normal data, with inverse-cdf weighting."
    )
    assert np.all(fixed_gaussian_cdf_cost < fixed_empirical_cdf_cost), (
        "Gaussian CDF cost should be lower than empirical CDF cost,"
        " on standard normal data, with inverse-cdf weighting."
    )


# %%
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
        approx_cost_cache[-1, :] - approx_cost_cache[0, :]
    ) / len(xs)
    np.testing.assert_array_equal(
        xs_edf_eval_point_quantiles,
        np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
    )

    edf_eval_points_2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    approx_cost_cache_2 = make_cumulative_edf_cache(xs, edf_eval_points_2)
    xs_edf_eval_point_quantiles_2 = (
        approx_cost_cache_2[-1, :] - approx_cost_cache_2[0, :]
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
    approx_cost = approximate_mle_edf_cost(
        xs, segment_starts, segment_ends, num_approx_quantiles
    )
    direct_cost = direct_mle_edf_cost(xs, segment_starts, segment_ends)
    approx_cost_cache = make_approximate_mle_edf_cost_cache(xs, num_approx_quantiles)
    pre_cached_approx_cost = pre_cached_approximate_mle_edf_cost(
        approx_cost_cache, segment_starts, segment_ends
    )
    print(f"Direct cost: {direct_cost}")
    print(f"Approximate cost: {approx_cost}")
    print(f"Pre-cached approximate cost: {pre_cached_approx_cost}")

    # Add more tests with different data and changepoints
    xs2 = np.array([1, 1, 2, 2, 3, 3])
    segment_starts2 = np.array([0, 3])
    segment_ends2 = np.array([3, 6])
    k2 = 2
    approx_cost2 = approximate_mle_edf_cost(xs2, segment_starts2, segment_ends2, k2)
    approx_cost_cache_2 = make_approximate_mle_edf_cost_cache(xs2, k2)
    pre_cached_approx_cost2 = pre_cached_approximate_mle_edf_cost(
        approx_cost_cache_2, segment_starts2, segment_ends2
    )
    direct_cost2 = direct_mle_edf_cost(xs2, segment_starts2, segment_ends2)

    print(f"Direct cost for xs2: {direct_cost2}")
    print(f"Approximate cost for xs2: {approx_cost2}")
    print(f"Pre-cached approximate cost for xs2: {pre_cached_approx_cost2}")

    no_change_cuts = np.array([[0, len(xs)]])
    no_change_starts = no_change_cuts[:, 0]
    no_change_ends = no_change_cuts[:, 1]
    no_change_approx_cost = approximate_mle_edf_cost(
        xs, no_change_starts, no_change_ends, num_approx_quantiles
    )
    no_change_approx_cost_cache = make_approximate_mle_edf_cost_cache(
        xs, num_approx_quantiles
    )
    no_change_approx_cost_pre_cached = pre_cached_approximate_mle_edf_cost(
        no_change_approx_cost_cache, no_change_starts, no_change_ends
    )
    no_change_direct_cost = direct_mle_edf_cost(xs, no_change_starts, no_change_ends)
    print(f"No change direct cost: {no_change_direct_cost}")
    print(f"No change approximate cost: {no_change_approx_cost}")
    print(f"No change pre-cached approximate cost: {no_change_approx_cost_pre_cached}")


@pytest.mark.parametrize(
    ["tolerance", "n_samples"], [(0.10, 60), (0.05, 120), (0.025, 1000)]
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

    one_change_approx_costs = approximate_mle_edf_cost(
        xs, correct_segment_starts, correct_segment_ends, num_quantiles=num_quantiles
    )
    one_change_direct_costs = direct_mle_edf_cost(
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

    single_segment_approx_cost = approximate_mle_edf_cost(
        xs,
        no_change_segment_starts,
        no_change_segment_ends,
        num_quantiles=num_quantiles,
    )
    single_segment_direct_cost = direct_mle_edf_cost(
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
    approx_costs = approximate_mle_edf_cost(
        xs, correct_segment_starts, correct_segment_ends, num_quantiles=num_quantiles
    )
    approx_cost_cache = make_approximate_mle_edf_cost_cache(xs, num_quantiles)
    precached_costs = pre_cached_approximate_mle_edf_cost(
        approx_cost_cache, correct_segment_starts, correct_segment_ends
    )
    np.testing.assert_allclose(approx_costs, precached_costs, rtol=rel_tol)

    # Compare on single segment
    single_approx_cost = approximate_mle_edf_cost(
        xs,
        no_change_segment_starts,
        no_change_segment_ends,
        num_quantiles=num_quantiles,
    )
    single_precached_cost = pre_cached_approximate_mle_edf_cost(
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
    direct_cost = direct_mle_edf_cost(
        xs, per_hundred_step_segment_starts, per_hundred_step_segment_ends
    )
    start_time = time.perf_counter()
    direct_cost = direct_mle_edf_cost(
        xs, per_hundred_step_segment_starts, per_hundred_step_segment_ends
    )
    end_time = time.perf_counter()
    direct_cost_eval_time = end_time - start_time
    total_direct_cost = np.sum(direct_cost)

    # print(
    #     f"Total direct cost: {total_direct_cost},"
    #     f" Time taken: {direct_cost_eval_time:.4e} seconds"
    # )
    assert direct_cost_eval_time < 5.0e-2, (
        "Direct evaluation time should be less than 0.05 seconds."
    )

    ### Approximate evaluation:
    # Call once in case of JIT compilation overhead:
    approximate_mle_edf_cost(
        xs,
        per_hundred_step_segment_starts,
        per_hundred_step_segment_ends,
        num_quantiles=num_approx_quantiles,
    )
    start_time = time.perf_counter()
    approx_cost = approximate_mle_edf_cost(
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
    approx_cost_cache = make_approximate_mle_edf_cost_cache(xs, num_approx_quantiles)
    pre_cached_approximate_mle_edf_cost(
        approx_cost_cache,
        per_hundred_step_segment_starts,
        per_hundred_step_segment_ends,
    )

    cache_start_time = time.perf_counter()
    approx_cost_cache = make_approximate_mle_edf_cost_cache(xs, num_approx_quantiles)
    cache_end_time = time.perf_counter()
    cache_creation_time = cache_end_time - cache_start_time

    start_time = time.perf_counter()
    pre_cached_cost = pre_cached_approximate_mle_edf_cost(
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


@pytest.mark.parametrize("apply_continuity_correction", [True, False])
def test_difficult_case(apply_continuity_correction: bool):
    # fmt: off
    signal = np.array(
        [
           -4.16757847e-01, -5.62668272e-02, -2.13619610e00,   1.64027081e00,
           -1.79343559e00,  -8.41747366e-01,  5.02881417e-01, -1.24528809e00,
           -1.05795222e00,  -9.09007615e-01,  5.51454045e-01,  2.29220801e00,
            4.15393930e-02, -1.11792545e00,   5.39058321e-01, -5.96159700e-01,
           -1.91304965e-02,  1.17500122e00,  -7.47870949e-01,  9.02525097e-03,
           -8.78107893e-01, -1.56434170e-01,  2.56570452e-01, -9.88779049e-01,
           -3.38821966e-01, -2.36184031e-01, -6.37655012e-01, -1.18761229e00,
           -1.42121723e00,  -1.53495196e-01, -2.69056960e-01,  2.23136679e00,
           -2.43476758e00,   1.12726505e-01,  3.70444537e-01,  1.35963386e00,
            5.01857207e-01, -8.44213704e-01,  9.76147160e-06,  5.42352572e-01,
           -3.13508197e-01,  7.71011738e-01, -1.86809065e00,   1.73118467e00,
            1.46767801e00,  -3.35677339e-01,  6.11340780e-01,  4.79705919e-02,
           -8.29135289e-01,  8.77102184e-02,  1.00036589e00,  -3.81092518e-01,
           -3.75669423e-01, -7.44707629e-02,  4.33496330e-01,  1.27837923e00,
           -6.34679305e-01,  5.08396243e-01,  2.16116006e-01, -1.85861239e00,
           -4.19316482e-01, -1.32328898e-01, -3.95702397e-02,  3.26003433e-01,
           -2.04032305e00,   4.62555231e-02, -6.77675577e-01, -1.43943903e00,
            5.24296430e-01,  7.35279576e-01, -6.53250268e-01,  8.42456282e-01,
           -3.81516482e-01,  6.64890091e-02, -1.09873895e00,   1.58448706e00,
           -2.65944946e00,  -9.14526229e-02,  6.95119605e-01, -2.03346655e00,
           -1.89469265e-01, -7.72186654e-02,  8.24703005e-01,  1.24821292e00,
           -4.03892269e-01, -1.38451867e00,   1.36723542e00,   1.21788563e00,
           -4.62005348e-01,  3.50888494e-01,  3.81866234e-01,  5.66275441e-01,
            2.04207979e-01,  1.40669624e00,  -1.73795950e00,   1.04082395e00,
            3.80471970e-01, -2.17135269e-01,  1.17353150e00,  -2.34360319e00,
            1.61615215e01,   1.53860780e01,   1.38668667e01,   1.54330926e01,
            1.46959136e01,   1.75852949e01,   1.68353327e01,   1.54406899e01,
            1.42807462e01,   1.44165854e01,   1.46749504e01,   1.44397655e01,
            1.40977539e01,   1.44090277e01,   1.47238205e01,   1.44831161e01,
            1.43014101e01,   1.40711081e01,   1.75504382e01,   1.35268268e01,
            1.39785853e01,   1.54323957e01,   1.46764199e01,   1.54238247e01,
            1.57991800e01,   1.62626137e01,   1.57519648e01,   1.40062390e01,
            1.61091433e01,   1.32350823e01,   1.48855787e01,   1.45018258e01,
            1.39392010e01,   1.55916665e01,   1.48167434e01,   1.60198547e01,
            1.35175345e01,   1.58463119e01,   1.54979401e01,   1.51265042e01,
            1.35811894e01,   1.47482259e01,   1.34533254e01,   1.29173481e01,
            1.82797454e01,   1.59708613e01,   1.67925929e01,   1.45709867e01,
            1.56961980e01,   1.56974163e01,   1.56015158e01,   1.50036595e01,
            1.47717524e01,   1.29303877e01,   1.56101441e01,   1.54234969e01,
            1.61178867e01,   1.47257579e01,   1.67418122e01,   1.45524991e01,
            1.37445728e01,   1.59381637e01,   1.45316537e01,   1.37452797e01,
            1.51248236e01,   1.57565021e01,   1.52414396e01,   1.54974256e01,
            1.91086926e01,   1.58211209e01,   1.65317603e01,   1.30141542e01,
            1.53650535e01,   1.57740820e01,   1.46355209e01,   1.41240205e01,
            1.53965202e01,   1.46853826e01,   1.44062444e01,   1.61495006e01,
            1.63355662e01,   1.53026293e01,   1.45457721e01,   1.55143707e01,
            1.58294584e01,   1.56306220e01,   1.35466357e01,   1.46619822e01,
            1.53591333e01,   1.56222204e01,   1.59607819e01,   1.57583703e01,
            1.38656815e01,   1.42925791e01,   1.37785708e01,   1.68044766e01,
            1.51804098e01,   1.55531643e01,   1.60330291e01,   1.46709976e01,
        ]
    ).reshape(-1)
    # fmt: on

    left_intervals = np.array([[0, 12], [1, 13], [2, 14], [3, 15], [4, 16]])
    right_intervals = np.array([[12, 24], [13, 25], [14, 26], [15, 27], [16, 28]])
    full_intervals = np.array([[0, 24], [1, 25], [2, 26], [3, 27], [4, 28]])

    left_direct_costs = direct_mle_edf_cost(
        signal,
        left_intervals[:, 0],
        left_intervals[:, 1],
        apply_continuity_correction=apply_continuity_correction,
    )
    right_direct_costs = direct_mle_edf_cost(
        signal,
        right_intervals[:, 0],
        right_intervals[:, 1],
        apply_continuity_correction=apply_continuity_correction,
    )
    no_change_direct_costs = direct_mle_edf_cost(
        signal,
        full_intervals[:, 0],
        full_intervals[:, 1],
        apply_continuity_correction=apply_continuity_correction,
    )

    direct_change_scores = no_change_direct_costs - (
        left_direct_costs + right_direct_costs
    )
    print(f"Change scores for difficult case: {direct_change_scores}")
    if apply_continuity_correction:
        assert np.all(direct_change_scores < 0), (
            "Change scores are all negative in this case with continuity correction."
        )
    else:
        assert np.all(direct_change_scores >= 0), (
            "Change scores should be non-negative in this case."
        )

    # Suggested value based on the length of data
    num_approx_quantiles = np.ceil(4 * np.log(len(signal)))

    left_approx_costs = approximate_mle_edf_cost(
        signal,
        left_intervals[:, 0],
        left_intervals[:, 1],
        num_quantiles=num_approx_quantiles,
        apply_continuity_correction=apply_continuity_correction,
    )
    right_approx_costs = approximate_mle_edf_cost(
        signal,
        right_intervals[:, 0],
        right_intervals[:, 1],
        num_quantiles=num_approx_quantiles,
        apply_continuity_correction=apply_continuity_correction,
    )
    no_change_approx_costs = approximate_mle_edf_cost(
        signal,
        full_intervals[:, 0],
        full_intervals[:, 1],
        num_quantiles=num_approx_quantiles,
        apply_continuity_correction=apply_continuity_correction,
    )
    approx_change_scores = no_change_approx_costs - (
        left_approx_costs + right_approx_costs
    )
    print(f"Approximate change scores for difficult case: {approx_change_scores}")
    if apply_continuity_correction:
        assert np.all(approx_change_scores < 0), (
            "Approximate change scores should be negative in this case with "
            "continuity correction."
        )
    else:
        assert np.all(approx_change_scores >= 0), (
            "Approximate change scores should be non-negative in this case."
        )

    cumulative_edf_cache = make_approximate_mle_edf_cost_cache(
        signal, num_quantiles=num_approx_quantiles
    )
    pre_cached_left_approx_costs = pre_cached_approximate_mle_edf_cost(
        cumulative_edf_cache,
        left_intervals[:, 0],
        left_intervals[:, 1],
        apply_continuity_correction=apply_continuity_correction,
    )
    pre_cached_right_approx_costs = pre_cached_approximate_mle_edf_cost(
        cumulative_edf_cache,
        right_intervals[:, 0],
        right_intervals[:, 1],
        apply_continuity_correction=apply_continuity_correction,
    )
    pre_cached_no_change_approx_costs = pre_cached_approximate_mle_edf_cost(
        cumulative_edf_cache,
        full_intervals[:, 0],
        full_intervals[:, 1],
        apply_continuity_correction=apply_continuity_correction,
    )
    pre_cached_approx_change_scores = pre_cached_no_change_approx_costs - (
        pre_cached_left_approx_costs + pre_cached_right_approx_costs
    )
    print(
        f"Pre-cached approximate change scores for difficult case: "
        f"{pre_cached_approx_change_scores}"
    )
    if apply_continuity_correction:
        assert np.all(pre_cached_approx_change_scores < 0), (
            "Pre-cached approximate change scores should be negative in this case with "
            "continuity correction."
        )
    else:
        assert np.all(pre_cached_approx_change_scores >= 0), (
            "Pre-cached approximate change scores should be non-negative in this case."
        )
