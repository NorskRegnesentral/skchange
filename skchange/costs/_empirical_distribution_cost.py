import numpy as np

from skchange.costs.base import BaseCost
from skchange.utils.numba import njit, numba_available
from skchange.utils.numba.stats import row_cumsum


def direct_empirical_distribution_cost(
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


@njit
def evaluate_edf_of_sorted_data(
    sorted_data: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """
    Evaluate empirical distribution function (EDF) from sorted data.

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


def evaluate_empirical_distribution_function(
    data: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the empirical distribution function (EDF) of a segment of data.

    Parameters
    ----------
    data : np.ndarray
        The data segment for which to compute the EDF.
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


def approximate_empirical_distribution_cost(
    xs: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    num_quantiles: int,
    apply_continuity_correction: bool = False,
) -> np.ndarray:
    """Compute approximate empirical distribution cost.

    Using `num_quantiles` quantile values to approximate the empirical distribution cost
    for a sequence `xs` with given segment cuts. The cost is computed as the integrated
    log-likelihood of the empirical distribution function (EDF), approximated by
    evaluating the EDF at `num_quantiles` specific quantiles.

    Parameters
    ----------
    xs : np.ndarray
        The first sequence.
    cuts : np.ndarray
        The cut intervals to consider.
    num_quantiles : int
        The number of terms to use in the approximation.

    Returns
    -------
    np.ndarray
        1D array of empirical distribution costs for each segment defined by `cuts`.
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

    segment_costs = np.zeros(len(segment_starts), dtype=np.float64)
    for i, (segment_start, segment_end) in enumerate(zip(segment_starts, segment_ends)):
        segment_length = segment_end - segment_start
        if segment_length <= 0:
            raise ValueError("Invalid segment length.")

        segment_data = xs[segment_start:segment_end]
        segment_edf_at_quantiles = evaluate_empirical_distribution_function(
            segment_data,
            quantile_values,
        )

        if apply_continuity_correction:
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

        # Store the negative integrated log-likelihood as the cost:
        segment_costs[i] = -segment_ll_at_mle

    return segment_costs


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

    full_sample_quantiles = 1.0 / (
        1 + np.exp(c * ((2 * np.arange(1, num_quantiles + 1) - 1) / num_quantiles - 1))
    )

    # Computing quantiles (which involves sorting) is slow with Numba:
    full_sample_quantile_values = np.quantile(xs, full_sample_quantiles)

    # Create a cache for cumulative empirical distribution function (EDF) values:
    cumulative_edf_quantiles = make_cumulative_edf_cache(
        xs, full_sample_quantile_values
    )

    return cumulative_edf_quantiles


@njit
def make_cumulative_edf_cache(
    xs: np.ndarray, quantile_values: np.ndarray
) -> np.ndarray:
    """Create a cache for cumulative empirical distribution function (EDF) values.

    This function computes the cumulative counts of values less than or equal to each
    quantile value in the input array `xs`. It returns a 2D array where each row
    corresponds to a quantile value and contains the cumulative counts of values
    less than or equal to that quantile.

    Parameters
    ----------
    xs : np.ndarray
        The input data array.
    quantile_values : np.ndarray
        The quantile values at which to compute the cumulative counts.

    Returns
    -------
    np.ndarray
        A 2D array where each row corresponds to a quantile value and contains the
        cumulative counts of values less than or equal to that quantile.
    """
    # Cache cumulative sums of the empirical distribution function (EDF):
    lte_quantile_value_mask = (xs[None, :] < quantile_values[:, None]).astype(
        np.float64
    )

    # Add 0.5 to the count for samples equal to the quantile values:
    lte_quantile_value_mask += 0.5 * (xs[None, :] == quantile_values[:, None])

    # Add the cumulative counts of values less than or equal each quantile:
    if numba_available:
        cumulative_edf_quantiles = row_cumsum(lte_quantile_value_mask, init_zero=True)
    else:
        # Fallback to numpy if Numba is not available:
        np.cumsum(
            lte_quantile_value_mask,
            axis=1,
            dtype=np.float64,
            out=lte_quantile_value_mask,
        )
        cumulative_edf_quantiles = lte_quantile_value_mask

        # Concatenate a column of zeros at the start to handle the first sample:
        cumulative_edf_quantiles = np.hstack(
            (np.zeros((len(quantile_values), 1)), cumulative_edf_quantiles)
        )

    return cumulative_edf_quantiles


@njit
def pre_cached_k_term_empirical_distribution_cost_approximation(
    cumulative_edf_quantiles: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    apply_continuity_correction: bool = False,
) -> np.ndarray:
    """
    Compute approximate empirical distribution cost from cumulative edf quantiles.

    Using `num_quantiles` quantile values to approximate the empirical distribution cost
    for a sequence `xs` with given segment cuts. The cost is computed as the integrated
    log-likelihood of the empirical distribution function (EDF), approximated by
    evaluating the EDF at `num_quantiles` specific quantiles.

    Parameters
    ----------
    cumulative_edf_quantiles : np.ndarray
        A 2D array containing the cumulative empirical distribution function values
        for the entire dataset, pre-computed from `make_edf_cost_approximation_cache`.
    segment_starts : np.ndarray
        The start indices of the segments.
    segment_ends : np.ndarray
        The end indices of the segments.

    Returns
    -------
    np.ndarray
        A 1D array of empirical distribution costs for each segment defined by `cuts`.
    """
    num_quantiles, n_samples = cumulative_edf_quantiles.shape

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

    segment_costs = np.zeros(len(segment_starts), dtype=np.float64)

    # Iterate over each segment defined by the starts and ends:
    for i, (segment_start, segment_end) in enumerate(zip(segment_starts, segment_ends)):
        segment_length = segment_end - segment_start
        if segment_length <= 0:
            raise ValueError("Invalid segment length.")

        # Shifted by 1 to account for the column of zeros added earlier:
        segment_edf_at_quantiles = (
            cumulative_edf_quantiles[:, 1 + segment_end - 1]
            - cumulative_edf_quantiles[:, 1 + segment_start - 1]
        ) / segment_length

        if apply_continuity_correction:
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

        # Store the negative log-likelihood as the cost:
        segment_costs[i] = -segment_ll_at_mle

    return segment_costs


class EmpiricalDistributionCost(BaseCost):
    """
    Empirical Distribution Cost.

    This cost function computes an approximate empirical distribution cost.
    It uses the integrated log-likelihood of the empirical distribution function (EDF)
    to evaluate the cost for each segment defined by the cuts.

    Parameters
    ----------
    param : any, optional (default=None)
        If None, the cost is evaluated for an interval-optimised parameter, often the
        maximum likelihood estimate. If not None, the cost is evaluated for the
        specified fixed parameter.
    use_cache : bool, optional (default=True)
        If True, precompute the empirical distribution function for faster evaluation.
    num_approximation_quantiles : int or None, optional (default=None)
        Number of quantiles to use for approximating the empirical distribution.
        If None, it will be set to 4 * log(n_samples) during fitting.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "distribution_type": "None",
        "is_conditional": False,
        "is_aggregated": False,
        "supports_fixed_param": False,
    }

    def __init__(
        self,
        param=None,
        use_cache=True,
        num_approximation_quantiles=None,
        n_samples_approximation_threshold=250,
    ):
        # Initialize the base class
        super().__init__(param)

        # Store parameters:
        self.use_cache = use_cache
        self.num_approximation_quantiles = num_approximation_quantiles
        self.n_samples_approximation_threshold = n_samples_approximation_threshold

        self.num_approximation_quantiles_ = None  # Will be set during fitting
        self._edf_cache = None  # Cache for empirical distribution function

    def _fit(self, X: np.ndarray, y=None):
        """Fit the cost.

        Precomputes cumulative empirical distribution function if caching is enabled.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y: None
            Ignored. Included for API consistency by convention.
        """
        self._param = self._check_param(self.param, X)
        n_samples = X.shape[0]

        if self.use_cache and n_samples >= self.n_samples_approximation_threshold:
            # Calculate num_approximation_quantiles if not provided
            if self.num_approximation_quantiles is None:
                self.num_approximation_quantiles_ = np.ceil(4 * np.log(n_samples))
            else:
                self.num_approximation_quantiles_ = self.num_approximation_quantiles

            # Compute empirical distribution function cache for each column of data:
            self._edf_cache = [
                make_edf_cost_approximation_cache(
                    self._X[:, col], self.num_approximation_quantiles_
                )
                for col in range(self._X.shape[1])
            ]

        return self

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost using empirical distribution.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval.
        """
        n_intervals = len(starts)
        n_samples, n_cols = self._X.shape

        costs = np.zeros((n_intervals, n_cols))

        for col in range(n_cols):
            if n_samples < self.n_samples_approximation_threshold:
                # For small datasets, compute the empirical distribution
                # cost directly, as it is more accurate and fast enough:
                costs[:, col] = direct_empirical_distribution_cost(
                    self._X[:, col],
                    segment_starts=starts,
                    segment_ends=ends,
                )
            elif self.use_cache:
                # Use precomputed cumulative EDF values from the cache
                costs[:, col] = (
                    pre_cached_k_term_empirical_distribution_cost_approximation(
                        self._edf_cache[col], segment_starts=starts, segment_ends=ends
                    )
                )
            else:
                costs[:, col] = approximate_empirical_distribution_cost(
                    self._X[:, col],
                    segment_starts=starts,
                    segment_ends=ends,
                    num_quantiles=self.num_approximation_quantiles_,
                )

        return costs

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate.

        Returns
        -------
        int
            The minimum valid size of an interval to evaluate.
        """
        # For EDF, we need at least 2 samples to compute a meaningful distribution
        if self.num_approximation_quantiles is not None:
            return self.num_approximation_quantiles + 2
        else:
            # If not fitted and no n_quantiles specified, assume a default:
            # For 12 data points, we get 10 quantiles, (ceil(4 * log(12)) = 10),
            # so we respect the requirement of having at least 2 samples
            # more than the number of quantiles.
            return 12

    def get_model_size(self, p: int) -> int:
        """Get the number of parameters in the cost function.

        Parameters
        ----------
        p : int
            Number of variables in the data.

        Returns
        -------
        int
            Number of parameters in the cost function.
        """
        # For the EDF, the model size depends on the number of quantiles?
        if self.is_fitted and self.num_approximation_quantiles_ is not None:
            return self.num_approximation_quantiles_ + 2
        elif self.num_approximation_quantiles is not None:
            return self.num_approximation_quantiles + 2
        else:
            # If not fitted or no num_approximation_quantiles specified,
            # assume a default value of 10 quantiles:
            return 10 + 2

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
        """
        # Define two different parameter sets for testing
        params1 = {"use_cache": True, "num_approximation_quantiles": 10}

        params2 = {"use_cache": False, "num_approximation_quantiles": None}

        return [params1, params2]
