# %%
import numpy as np

from skchange.utils.numba import njit  # , numba_available

numba_available = False


# %%
def direct_empirical_distribution_cost(xs: np.ndarray, cuts: np.ndarray) -> float:
    """
    Compute the empirical distribution cost for a sequence of values
    given a set of changepoints.

    Parameters
    ----------
    xs : np.ndarray
        The input data array.
    cuts : np.ndarray
        The cut intervals to evaluate the empirical distribution cost on.

    Returns
    -------
    float
        The total empirical distribution cost.
    """
    integrated_ll_at_mle = 0.0

    edf_eval_points = np.sort(xs)[1:-1]  # Exclude the first and last samples
    segment_edf_per_sample = np.zeros(len(xs) - 2, dtype=np.float64)
    reciprocal_full_data_cdf_weights = len(xs) / (
        np.arange(2, len(xs), dtype=np.float64)
        * np.arange(len(xs) - 1, 1, -1, dtype=np.float64)
    )

    segment_starts = cuts[:, 0]
    segment_ends = cuts[:, 1]

    for segment_start, segment_end in zip(segment_starts, segment_ends):
        segment_length = segment_end - segment_start
        segment_data = xs[segment_start:segment_end]

        # segment_edf_per_sample[:] = evaluate_edf_of_data_segment(
        #     segment_data,
        #     edf_eval_points,
        # )
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

        integrated_ll_at_mle += segment_length * (
            np.sum(
                (
                    segment_edf_per_sample * np.log(segment_edf_per_sample)
                    + one_minus_segment_empirical_distribution_per_sample
                    * np.log(one_minus_segment_empirical_distribution_per_sample)
                )
                * reciprocal_full_data_cdf_weights
            )
        )

    return -integrated_ll_at_mle


def compute_segment_edf_at_ordered_global_samples(
    segment: np.ndarray,
    evaluation_points: np.ndarray,
) -> np.ndarray:
    """
    Computes the empirical distribution function (EDF) of a data segment,
    evaluated at the ordered samples of the entire dataset (excluding the
    first and last).

    This corresponds to F_kl = F_{segment_k}(X_(l)) from the paper
    "Nonparametric Maximum Likelihood Approach to Multiple Change-Point Problems"
    by Zou et al. (2014), where X_(l) are the order statistics of the
    entire dataset, for l = 2, ..., n-1 (using 1-based indexing for l as in
    the paper, which means data sorted at indices 1 to n-2 for 0-indexed arrays).

    Parameters
    ----------
    segment : np.ndarray
        The data segment for which to compute the EDF.
    evaluation_points : np.ndarray
        The points at which to evaluate the segment EDF.

    Returns
    -------
    np.ndarray
        An array of EDF values. The i-th value is the EDF of the segment
        evaluated at `np.sort(data)[i+1]`.
        Returns an empty array if `len(data) < 3` (as there are no
        evaluation points from the second to second-to-last sample)
        or if the segment is empty.
    """
    if evaluation_points.size == 0:
        return np.array([])

    segment_len = len(segment)

    if segment_len == 0:
        # EDF of an empty segment.
        # The paper implies segments have positive length.
        # Return an empty array, or an array of zeros with length of evaluation_points,
        # depending on desired behavior. Empty array is consistent.
        return np.array([])

    # Sort the segment for efficient counting using searchsorted
    sorted_segment = np.sort(segment)

    # For each evaluation_point, count how many elements in sorted_segment
    # are less than or equal to it.
    # np.searchsorted(a, v, side='right') returns an index i such that all a[:i] <= v
    # and all a[i:] > v. If a is sorted, this is the count of elements <= v.
    counts = np.searchsorted(sorted_segment, evaluation_points, side="right")

    edf_values = counts / segment_len

    return edf_values


def evaluate_empirical_distribution_function(
    data: np.ndarray,
    values: np.ndarray,
):
    """
    Evaluate the empirical distribution function (EDF) of a segment of data
    at given values.

    Parameters
    ----------
    sorted_data : np.ndarray
        The sorted data segment.
    values : np.ndarray
        The values at which to evaluate the EDF.

    Returns
    -------
    np.ndarray
        The EDF values at the specified points.
    """
    sorted_data = np.sort(data)

    segment_edf_values = evaluate_edf_of_sorted_data(
        sorted_data,
        values,
    )

    return segment_edf_values


def evaluate_edf_of_data_segment(
    data: np.ndarray,
    values: np.ndarray,
):
    """
    Evaluate the empirical distribution function (EDF) of a segment of data
    at given values.

    Parameters
    ----------
    sorted_data : np.ndarray
        The sorted data segment.
    values : np.ndarray
        The values at which to evaluate the EDF.

    Returns
    -------
    np.ndarray
        The EDF values at the specified points.
    """
    ### NOTE: Sorting within the njit function is really slow!
    sorted_data = np.sort(data)
    if len(sorted_data) < 2:
        raise ValueError("Data segment must contain at least two elements.")

    # Use searchsorted to find indices where each value would fit
    # in the sorted data. Effectively counts how many elements in
    # sorted_data are less than or equal to each value in values,
    # without a "continuity correction" as in the cached version.
    indices = np.searchsorted(sorted_data, values, side="right")

    # Normalize the counts to get the EDF values:
    segment_edf_values = indices / len(sorted_data)

    return segment_edf_values


@njit
def evaluate_edf_of_sorted_data(
    sorted_data: np.ndarray,
    values: np.ndarray,
):
    """
    Evaluate the empirical distribution function (EDF) of a segment of data
    at given values.

    Parameters
    ----------
    sorted_data : np.ndarray
        The sorted data segment.
    values : np.ndarray
        The values at which to evaluate the EDF.

    Returns
    -------
    np.ndarray
        The EDF values at the specified points.
    """
    if len(sorted_data) < 2:
        raise ValueError("Data segment must contain at least two elements.")

    # Use searchsorted to find indices where each value would fit in the sorted data:
    # Effectively counts how many elements in sorted_data are less than or equal to
    # each value in values, without the "continuity correction" that is applied
    # in the cached version.
    indices = np.searchsorted(sorted_data, values, side="right")

    # Normalize the counts to get the EDF values:
    segment_edf_values = indices / len(sorted_data)

    return segment_edf_values


# %%
def k_term_empirical_distribution_cost_approximation(
    xs: np.ndarray, cuts: np.ndarray, num_quantiles: int
) -> float:
    """
    Compute an approximate empirical distribution cost for a sequence of values
    given a set of changepoints, using a k-term approximation.

    Parameters
    ----------
    xs : np.ndarray
        The first sequence.
    changepoints : np.ndarray
        The changepoints to consider.
    num_quantiles : int
        The number of terms to use in the approximation.

    Returns
    -------
    float
        The total empirical distribution cost approximation.
    """
    # This is a placeholder for the actual implementation.
    # In practice, this function would compute the cost based on the
    # first num_quantiles terms.
    if num_quantiles <= 0:
        raise ValueError("num_quantiles must be a positive integer.")
    if num_quantiles > len(xs) - 2:
        raise ValueError(
            "num_quantiles should not be greater than the number of samples minus 2."
        )

    n_samples = len(xs)
    c = -np.log(2 * n_samples - 1)  # Constant term for the approximation

    quantiles = 1.0 / (
        1 + np.exp(c * ((2 * np.arange(1, num_quantiles + 1) - 1) / num_quantiles - 1))
    )
    quantile_values = np.quantile(xs, quantiles)

    integrated_ll_at_mle = 0.0
    segment_starts = cuts[:, 0]
    segment_ends = cuts[:, 1]
    for segment_start, segment_end in zip(segment_starts, segment_ends):
        segment_length = segment_end - segment_start
        if segment_length <= 0:
            raise ValueError("Invalid segment length.")

        segment_data = xs[segment_start:segment_end]
        segment_edf_at_quantiles = evaluate_empirical_distribution_function(
            segment_data,
            quantile_values,
        )

        # Apply continuity correction:
        segment_edf_at_quantiles -= 1 / (2 * segment_length)

        # Clip to within (0, 1) to avoid log(0) issues:
        segment_edf_at_quantiles = np.clip(segment_edf_at_quantiles, 1e-10, 1 - 1e-10)
        one_minus_segment_edf_at_quantiles = 1 - segment_edf_at_quantiles

        segment_ll_at_mle = (
            (-2.0 * c / num_quantiles)
            * segment_length
            * (
                np.sum(
                    segment_edf_at_quantiles * np.log(segment_edf_at_quantiles)
                    + one_minus_segment_edf_at_quantiles
                    * np.log(one_minus_segment_edf_at_quantiles)
                )
            )
        )

        integrated_ll_at_mle += segment_ll_at_mle

    return -integrated_ll_at_mle


def cached_k_term_empirical_distribution_cost_approximation(
    xs: np.ndarray, cuts: np.ndarray, num_quantiles: int
) -> float:
    """
    Compute an approximate empirical distribution cost for a sequence of values
    given a set of changepoints, using a k-term approximation.

    Parameters
    ----------
    xs : np.ndarray
        The first sequence.
    changepoints : np.ndarray
        The changepoints to consider.
    num_quantiles : int
        The number of terms to use in the approximation.

    Returns
    -------
    float
        The total empirical distribution cost approximation.
    """
    # This is a placeholder for the actual implementation.
    # In practice, this function would compute the cost based on the
    # first num_quantiles terms.
    if num_quantiles <= 0:
        raise ValueError("num_quantiles must be a positive integer.")
    if num_quantiles > len(xs) - 2:
        raise ValueError(
            "num_quantiles should not be greater than the number of samples minus 2."
        )

    n_samples = len(xs)
    c = -np.log(2 * n_samples - 1)  # Constant term for the approximation

    quantiles = 1.0 / (
        1 + np.exp(c * ((2 * np.arange(1, num_quantiles + 1) - 1) / num_quantiles - 1))
    )
    quantile_values = np.quantile(xs, quantiles)

    # Cache cumulative sums of the empirical distribution function (EDF):
    full_sample_cumulative_edf = np.cumulative_sum(
        xs < quantile_values[:, None], axis=1
    ) + 0.5 * np.cumulative_sum(xs == quantile_values[:, None], axis=1)

    # Concatenate a column of zeros at the start to handle the first sample:
    full_sample_cumulative_edf = np.hstack(
        (np.zeros((len(quantile_values), 1)), full_sample_cumulative_edf)
    )

    integrated_ll_at_mle = 0.0
    segment_starts = cuts[:, 0]
    segment_ends = cuts[:, 1]
    for segment_start, segment_end in zip(segment_starts, segment_ends):
        segment_length = segment_end - segment_start
        if segment_length <= 0:
            raise ValueError("Invalid segment length.")

        # Shifted by 1 to account for the column of zeros added earlier:
        segment_edf_at_quantiles = (
            full_sample_cumulative_edf[:, 1 + segment_end - 1]
            - full_sample_cumulative_edf[:, 1 + segment_start - 1]
        ) / segment_length

        # Apply continuity correction:
        segment_edf_at_quantiles -= 1 / (2 * segment_length)

        # Clip to within (0, 1) to avoid log(0) issues:
        segment_edf_at_quantiles = np.clip(segment_edf_at_quantiles, 1e-10, 1 - 1e-10)
        one_minus_segment_edf_at_quantiles = 1 - segment_edf_at_quantiles

        segment_ll_at_mle = (
            (-2.0 * c / num_quantiles)
            * segment_length
            * (
                np.sum(
                    segment_edf_at_quantiles * np.log(segment_edf_at_quantiles)
                    + one_minus_segment_edf_at_quantiles
                    * np.log(one_minus_segment_edf_at_quantiles)
                )
            )
        )

        integrated_ll_at_mle += segment_ll_at_mle

    return -integrated_ll_at_mle


@njit
def row_cumulative_sum(arr: np.ndarray, inplace=False) -> np.ndarray:
    """
    Compute the cumulative sum of each row in a 2D array.

    Parameters
    ----------
    arr : np.ndarray
        The input 2D array.

    Returns
    -------
    np.ndarray
        A 2D array where each row contains the cumulative sum of the corresponding row ¨
        in the input array.
    """
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    if not inplace:
        arr = arr.copy()

    for row in range(arr.shape[0]):
        for col in range(1, arr.shape[1]):
            arr[row, col] += arr[row, col - 1]
        # arr[row, :] = np.cumsum(arr[row, :])

    return arr


def make_edf_cost_approximation_cache(xs: np.ndarray, num_quantiles: int) -> np.ndarray:
    """
    Create a cache for the empirical distribution function (EDF) cost approximation.

    Parameters
    ----------
    xs : np.ndarray
        The input data array.
    num_quantiles : int
        The number of quantiles to use in the approximation.

    Returns
    -------
    np.ndarray
        A 2D array where each row corresponds to a quantile and contains the cumulative
        EDF values for the entire dataset.
    """
    if num_quantiles <= 0:
        raise ValueError("num_quantiles must be a positive integer.")
    if num_quantiles > len(xs) - 2:
        raise ValueError(
            "num_quantiles should not be greater than the number of samples minus 2."
        )

    n_samples = len(xs)
    c = -np.log(2 * n_samples - 1)  # Constant term for the approximation

    quantiles = 1.0 / (
        1 + np.exp(c * ((2 * np.arange(1, num_quantiles + 1) - 1) / num_quantiles - 1))
    )

    # Computing quantiles (which involves sorting) is slow with Numba:
    quantile_values = np.quantile(xs, quantiles)

    # Cache cumulative sums of the empirical distribution function (EDF):

    lte_quantile_value_mask = (xs[None, :] < quantile_values[:, None]).astype(
        np.float64
    )

    # Add 0.5 to the count for samples equal to the quantile values:
    lte_quantile_value_mask += 0.5 * (xs[None, :] == quantile_values[:, None])

    # Add the cumulative counts of values less than or equal each quantile:
    if numba_available:
        full_sample_cumulative_edf = row_cumulative_sum(
            lte_quantile_value_mask, inplace=True
        )
    else:
        # Fallback to numpy if Numba is not available:
        np.cumulative_sum(
            lte_quantile_value_mask,
            axis=1,
            dtype=np.float64,
            out=lte_quantile_value_mask,
        )
        full_sample_cumulative_edf = lte_quantile_value_mask

    # Concatenate a column of zeros at the start to handle the first sample:
    full_sample_cumulative_edf = np.hstack(
        (np.zeros((len(quantile_values), 1)), full_sample_cumulative_edf)
    )

    return full_sample_cumulative_edf


@njit
def pre_cached_k_term_empirical_distribution_cost_approximation(
    n_samples: int,
    full_sample_cumulative_edf: np.ndarray,
    cuts: np.ndarray,
    num_quantiles: int,
) -> float:
    """
    Compute an approximate empirical distribution cost for a sequence of values
    given a set of changepoints, using a k-term approximation.

    Parameters
    ----------
    xs : np.ndarray
        The first sequence.
    changepoints : np.ndarray
        The changepoints to consider.
    num_quantiles : int
        The number of terms to use in the approximation.

    Returns
    -------
    float
        The total empirical distribution cost approximation.
    """
    # This is a placeholder for the actual implementation.
    # In practice, this function would compute the cost based on the
    # first num_quantiles terms.
    if num_quantiles <= 0:
        raise ValueError("num_quantiles must be a positive integer.")
    if num_quantiles > n_samples - 2:
        raise ValueError(
            "num_quantiles should not be greater than the number of samples minus 2."
        )

    c = -np.log(2 * n_samples - 1)  # Constant term for the approximation

    integrated_ll_at_mle = 0.0
    segment_starts = cuts[:, 0]
    segment_ends = cuts[:, 1]
    for segment_start, segment_end in zip(segment_starts, segment_ends):
        segment_length = segment_end - segment_start
        if segment_length <= 0:
            raise ValueError("Invalid segment length.")

        # Shifted by 1 to account for the column of zeros added earlier:
        segment_edf_at_quantiles = (
            full_sample_cumulative_edf[:, 1 + segment_end - 1]
            - full_sample_cumulative_edf[:, 1 + segment_start - 1]
        ) / segment_length

        # Apply continuity correction:
        segment_edf_at_quantiles -= 1 / (2 * segment_length)

        # Clip to within (0, 1) to avoid log(0) issues:
        segment_edf_at_quantiles = np.clip(segment_edf_at_quantiles, 1e-10, 1 - 1e-10)
        one_minus_segment_edf_at_quantiles = 1 - segment_edf_at_quantiles

        segment_ll_at_mle = (
            (-2.0 * c / num_quantiles)
            * segment_length
            * (
                np.sum(
                    segment_edf_at_quantiles * np.log(segment_edf_at_quantiles)
                    + one_minus_segment_edf_at_quantiles
                    * np.log(one_minus_segment_edf_at_quantiles)
                )
            )
        )

        integrated_ll_at_mle += segment_ll_at_mle

    return -integrated_ll_at_mle


# %% Benchmark the njit functions versus the non-njit versions:
xs = np.random.normal(size=100_000)
sorted_xs = np.sort(xs)
changepoints = np.array([0, len(xs) // 2, len(xs)])
quantile_values = np.quantile(xs, np.linspace(0, 1, 30)[1:-1])
# njit_cost = direct_empirical_distribution_cost(xs, changepoints)

# %% Benchmark the 'evaluate_edf_of_data_segment' function
# %%timeit
# edf_values = evaluate_edf_of_data_segment(xs, quantile_values)

# %% Benchmark the njit version of 'evaluate_edf_of_data_segment'
# %%timeit
# njit_edf_values = njit_evaluate_edf_of_data_segment(xs, quantile_values)

# %% Benchmark the 'njit_evaluate_edf_of_sorted_data' function
# %%timeit
# njit_sorted_edf_values = njit_evaluate_edf_of_sorted_data(sorted_xs, quantile_values)

# %% Benchmark the njit version
# %%timeit
# njit_cost = njit_k_term_empirical_distribution_cost_approximation(
#     xs, changepoints, k=30
# )

# %% Benchmark the non-njit version
# %%timeit
# direct_cost = k_term_empirical_distribution_cost_approximation(xs, changepoints, k=30)


# %% Test the direct empirical distribution cost function
def test_direct_empirical_distribution_cost():
    xs = np.array([1, 2, 3, 4, 5])
    cuts = np.array([[0, 2], [2, 5]])
    cost = direct_empirical_distribution_cost(xs, cuts)
    print(f"Cost: {cost}")

    # Add more tests with different data and changepoints
    xs2 = np.array([1, 1, 2, 2, 3, 3])
    cuts2 = np.array([[0, 3], [3, 6]])
    cost2 = direct_empirical_distribution_cost(xs2, cuts2)
    print(f"Cost for xs2: {cost2}")


# test_direct_empirical_distribution_cost()


# %%
def test_k_term_empirical_distribution_cost_approximation():
    xs = np.array([1, 2, 3, 4, 5])
    cuts = np.array([[0, 2], [2, 5]])
    k = 3
    approx_cost = k_term_empirical_distribution_cost_approximation(xs, cuts, k)
    direct_cost = direct_empirical_distribution_cost(xs, cuts)
    cached_approx_cost = cached_k_term_empirical_distribution_cost_approximation(
        xs, cuts, k
    )
    approx_cost_cache = make_edf_cost_approximation_cache(xs, k)
    pre_cached_approx_cost = (
        pre_cached_k_term_empirical_distribution_cost_approximation(
            len(xs), approx_cost_cache, cuts, k
        )
    )
    print(f"Direct cost: {direct_cost}")
    print(f"Approximate cost: {approx_cost}")
    print(f"Cached approximate cost: {cached_approx_cost}")
    print(f"Pre-cached approximate cost: {pre_cached_approx_cost}")

    # Add more tests with different data and changepoints
    xs2 = np.array([1, 1, 2, 2, 3, 3])
    cuts2 = np.array([[0, 3], [3, 6]])
    k2 = 2
    approx_cost2 = k_term_empirical_distribution_cost_approximation(xs2, cuts2, k2)
    cached_approx_cost2 = cached_k_term_empirical_distribution_cost_approximation(
        xs2, cuts2, k2
    )
    approx_cost_cache_2 = make_edf_cost_approximation_cache(xs2, k2)
    pre_cached_approx_cost2 = (
        pre_cached_k_term_empirical_distribution_cost_approximation(
            len(xs2), approx_cost_cache_2, cuts2, k2
        )
    )
    direct_cost2 = direct_empirical_distribution_cost(xs2, cuts2)

    print(f"Direct cost for xs2: {direct_cost2}")
    print(f"Approximate cost for xs2: {approx_cost2}")
    print(f"Cached approximate cost for xs2: {cached_approx_cost2}")
    print(f"Pre-cached approximate cost for xs2: {pre_cached_approx_cost2}")

    no_change_cuts = np.array([[0, len(xs)]])
    no_change_approx_cost = k_term_empirical_distribution_cost_approximation(
        xs, no_change_cuts, k
    )
    no_change_approx_cost_cached = (
        cached_k_term_empirical_distribution_cost_approximation(xs, no_change_cuts, k)
    )
    no_change_approx_cost_cache = make_edf_cost_approximation_cache(xs, k)
    no_change_approx_cost_pre_cached = (
        pre_cached_k_term_empirical_distribution_cost_approximation(
            len(xs), no_change_approx_cost_cache, no_change_cuts, k
        )
    )
    no_change_direct_cost = direct_empirical_distribution_cost(xs, no_change_cuts)
    print(f"No change direct cost: {no_change_direct_cost}")
    print(f"No change approximate cost: {no_change_approx_cost}")
    print(f"No change cached approximate cost: {no_change_approx_cost_cached}")
    print(f"No change pre-cached approximate cost: {no_change_approx_cost_pre_cached}")


def test_approximation_vs_direct_on_longer_data(n_samples):
    np.random.seed(42)  # For reproducibility
    first_segment = np.random.normal(size=n_samples)
    second_segment = np.random.normal(
        size=n_samples, loc=5
    )  # Shifted mean for the second segment
    xs = np.concatenate([first_segment, second_segment])

    correct_cuts = np.array([[0, n_samples], [n_samples, len(xs)]])
    no_change_cuts = np.array([[0, len(xs)]])

    k = int(4 * np.log(len(xs)))  # Example k value based on the length of xs

    print(f"Using k = {k} for longer data.")
    print(f"Number of samples: {len(xs)}")
    changepoint_approx_cost = k_term_empirical_distribution_cost_approximation(
        xs, correct_cuts, k
    )
    changepoint_approx_cost_cached = (
        cached_k_term_empirical_distribution_cost_approximation(xs, correct_cuts, k)
    )
    changepoint_direct_cost = direct_empirical_distribution_cost(xs, correct_cuts)
    print(f"Direct cost on longer data: {changepoint_direct_cost}")
    print(f"Approximate cost on longer data: {changepoint_approx_cost}")
    print(f"Cached approximate cost on longer data: {changepoint_approx_cost_cached}")

    single_segment_approx_cost = k_term_empirical_distribution_cost_approximation(
        xs, no_change_cuts, k
    )
    single_segment_approx_cost_cached = (
        cached_k_term_empirical_distribution_cost_approximation(xs, no_change_cuts, k)
    )
    single_segment_direct_cost = direct_empirical_distribution_cost(xs, no_change_cuts)
    print(f"Direct cost for single segment: {single_segment_direct_cost}")
    print(f"Approximate cost for single segment: {single_segment_approx_cost}")
    print(
        f"Cached approximate cost for single segment: {single_segment_approx_cost_cached}"
    )

    relative_error = (
        np.abs(
            (changepoint_approx_cost - changepoint_direct_cost)
            / changepoint_direct_cost
        )
        if changepoint_direct_cost != 0
        else np.inf
    )
    print(f"Relative error: {relative_error:.4f}")


# For really few samples, the approximation bad:
# test_k_term_empirical_distribution_cost_approximation()

# For longer data, the approximation is good (<10% relative error):
test_approximation_vs_direct_on_longer_data(100)  # (7.3% relative error)
test_approximation_vs_direct_on_longer_data(1000)  # (2.8% relative error)
test_approximation_vs_direct_on_longer_data(10000)  # (1.4% relative error)

# %% Benchmark cached vs non-cached approximation, when evaluating on the same data,
# with many cuts:
xs_2 = np.random.normal(size=100_000)
cuts_2 = np.array([[i * 1000, (i + 1) * 1000] for i in range(len(xs_2) // 1000)])

direct_cost_2 = direct_empirical_distribution_cost(xs_2, cuts_2)
cached_test_result = cached_k_term_empirical_distribution_cost_approximation(
    xs_2, cuts_2, num_quantiles=30
)

approx_test_result = k_term_empirical_distribution_cost_approximation(
    xs_2, cuts_2, num_quantiles=30
)

approx_cost_cache_2 = make_edf_cost_approximation_cache(xs_2, num_quantiles=30)
pre_cached_test_result = pre_cached_k_term_empirical_distribution_cost_approximation(
    len(xs_2), approx_cost_cache_2, cuts_2, num_quantiles=30
)

# %%
# %%timeit
make_edf_cost_approximation_cache(xs_2, num_quantiles=30)

# %%
# %%timeit
## NOTE: Making the cache is slow, but decreases evaluation time significantly.
pre_cached_k_term_empirical_distribution_cost_approximation(
    len(xs_2), approx_cost_cache_2, cuts_2, num_quantiles=30
)

# %%
# %%timeit
cached_k_term_empirical_distribution_cost_approximation(xs_2, cuts_2, num_quantiles=30)

# %%
# %%timeit
k_term_empirical_distribution_cost_approximation(xs_2, cuts_2, num_quantiles=30)


# %%
def benchmark_direct_vs_approximation():
    import time

    n_samples = 100_000
    xs = np.random.normal(size=n_samples)
    changepoints = np.array([0, n_samples // 2, n_samples])

    start_time = time.perf_counter()
    direct_cost = direct_empirical_distribution_cost(xs, changepoints)
    end_time = time.perf_counter()
    print(
        f"Direct cost: {direct_cost}, Time taken: {end_time - start_time:.4e} seconds"
    )

    k = int(4 * np.log(n_samples))
    start_time = time.perf_counter()
    approx_cost = k_term_empirical_distribution_cost_approximation(xs, changepoints, k)
    end_time = time.perf_counter()
    print(
        f"Approximate cost: {approx_cost}, Time taken: {end_time - start_time:.4e} sec."
    )


benchmark_direct_vs_approximation()

# %%
n_samples = 100_000
xs = np.random.normal(size=n_samples)
changepoints = np.array([0, n_samples // 2, n_samples])

# %%
# %%timeit
direct_cost = direct_empirical_distribution_cost(xs, changepoints)

# %%
# %%timeit
k = int(4 * np.log(n_samples))
approx_cost = k_term_empirical_distribution_cost_approximation(xs, changepoints, k)

# %%
