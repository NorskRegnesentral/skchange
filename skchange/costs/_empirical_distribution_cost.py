import numpy as np
from numpy.typing import ArrayLike

from skchange.costs.base import BaseCost
from skchange.utils.numba import njit, numba_available
from skchange.utils.numba.stats import col_cumsum


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


@njit
def compute_finite_difference_derivatives(ts: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Compute second-order finite difference derivatives.

    Without assuming uniform sampling, this function computes the second-order
    derivatives of y(t) using a finite difference approximation of the derivative.

    Parameters
    ----------
    ts : np.ndarray
        The sampling points at which to compute the derivatives. Assumed to be sorted.
    ys : np.ndarray
        The values of the function at the sampling points.

    Returns
    -------
    np.ndarray
        The approximated second-order derivatives of y(t) at the sampling points.
    """
    if len(ts) < 3:
        raise ValueError("At least three quantiles are required.")

    diff_weights = np.zeros((len(ts), len(ts)), dtype=np.float64)
    steps = ts[1:] - ts[:-1]

    # Second-order forward finite difference weights for the first quantile:
    first_steps_sum = steps[0] + steps[1]
    diff_weights[0, 0] = (-2 * steps[0] - steps[1]) / (steps[0] * first_steps_sum)
    diff_weights[0, 1] = first_steps_sum / (steps[0] * steps[1])
    diff_weights[0, 2] = -steps[0] / (steps[1] * first_steps_sum)

    # Central second-order finite difference weights:
    for i in range(1, len(ts) - 1):
        steps_sum = steps[i - 1] + steps[i]

        # For uniform steps, current_weight == 0.
        prev_weight = -steps[i] / (steps_sum * steps[i - 1])
        current_weight = (steps[i] - steps[i - 1]) / (steps[i] * steps[i - 1])
        next_weight = steps[i - 1] / (steps_sum * steps[i])

        diff_weights[i, i - 1] = prev_weight
        diff_weights[i, i] = current_weight
        diff_weights[i, i + 1] = next_weight

    # Second-order backward finite difference weights for the last quantile:
    last_steps_sum = steps[-2] + steps[-1]
    diff_weights[-1, -3] = (steps[-1]) / (steps[-2] * last_steps_sum)
    diff_weights[-1, -2] = (-last_steps_sum) / (steps[-2] * steps[-1])
    diff_weights[-1, -1] = (steps[-1] + last_steps_sum) / (steps[-1] * last_steps_sum)

    derivatives = np.dot(diff_weights, ys)

    return derivatives


def make_approximate_mle_edf_cost_quantile_points(
    X: np.ndarray, num_quantiles: int
) -> np.ndarray:
    """Compute quantile points for the approximation of EDF integral."""
    n_samples = X.shape[0]

    integrated_edf_scaling = -np.log(2 * n_samples - 1)
    quantiles_range = np.arange(1, num_quantiles + 1)
    integration_quantiles = 1.0 / (
        1
        + np.exp(
            integrated_edf_scaling * ((2 * quantiles_range - 1) / num_quantiles - 1)
        )
    )

    quantile_points = np.quantile(X, integration_quantiles, axis=0)

    return quantile_points


@njit
def make_fixed_cdf_cost_quantile_weights(
    fixed_quantiles: np.ndarray,
    quantile_points: np.ndarray,
) -> np.ndarray:
    """
    Cache the fixed CDF quantiles and their derivatives.

    This function computes the second-order finite difference derivatives of the
    empirical distribution function (EDF) at the specified fixed quantiles.

    Parameters
    ----------
    fixed_quantiles : np.ndarray
        The quantiles at which to compute the EDF.
    quantile_points : np.ndarray
        The pre-image of the fixed quantiles.

    Returns
    -------
    np.ndarray
        The quantile integration weights for the fixed CDF cost.
    """
    if len(fixed_quantiles) < 3:
        raise ValueError("At least three fixed quantile values are required.")

    reciprocal_fixed_cdf_weights = 1.0 / (fixed_quantiles * (1.0 - fixed_quantiles))
    fixed_quantile_derivatives = compute_finite_difference_derivatives(
        ts=quantile_points,
        ys=fixed_quantiles,
    )
    quantile_integration_weights = (
        reciprocal_fixed_cdf_weights * fixed_quantile_derivatives
    )

    return quantile_integration_weights


def fixed_cdf_empirical_distribution_cost_cached_edf(
    cumulative_edf_quantiles: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    log_fixed_quantiles: np.ndarray,
    log_one_minus_fixed_quantiles: np.ndarray,
    quantile_weights: np.ndarray,
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
    num_quantiles = cumulative_edf_quantiles.shape[1]

    # Compute the integrals weights by approximating the derivative of the fixed CDF:
    if num_quantiles < 3:
        raise ValueError("At least three fixed quantile values are required.")

    segment_costs = np.zeros(len(segment_starts), dtype=np.float64)
    segment_quantiles = np.zeros(num_quantiles, dtype=np.float64)

    for i, (segment_start, segment_end) in enumerate(zip(segment_starts, segment_ends)):
        segment_length = segment_end - segment_start

        segment_quantiles[:] = cumulative_edf_quantiles[1 + segment_end - 1, :]
        segment_quantiles -= cumulative_edf_quantiles[1 + segment_start - 1, :]
        segment_quantiles /= float(segment_length)

        if apply_continuity_correction:
            segment_quantiles -= 1 / (2 * segment_length)

        # Clip to avoid log(0) issues:
        segment_quantiles = np.clip(segment_quantiles, 1e-10, 1 - 1e-10)
        one_minus_segment_quantiles = 1.0 - segment_quantiles

        integrated_ll_at_fixed_cdf = segment_length * (
            np.sum(
                (
                    segment_quantiles * log_fixed_quantiles
                    + one_minus_segment_quantiles * log_one_minus_fixed_quantiles
                )
                * quantile_weights
            )
        )

        # The cost equals twice the negative integrated log-likelihood:
        segment_costs[i] = -2.0 * integrated_ll_at_fixed_cdf

    return segment_costs


def approximate_mle_edf_cost(
    xs: np.ndarray,
    quantile_points: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    segment_costs: np.ndarray | None = None,
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
    assert xs.ndim == 1, "Input data must be a 1D array."
    num_quantiles = len(quantile_points)

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
    integrated_edf_scaling = -np.log(
        2 * n_samples - 1
    )  # Constant term for the approximation

    if segment_costs is None:
        # Initialize segment costs if not provided:
        segment_costs = np.zeros(len(segment_starts), dtype=np.float64)

    for i, (segment_start, segment_end) in enumerate(zip(segment_starts, segment_ends)):
        segment_length = segment_end - segment_start
        if segment_length <= 0:
            raise ValueError("Invalid segment length.")

        segment_data = xs[segment_start:segment_end]
        segment_edf_at_quantiles = evaluate_empirical_distribution_function(
            segment_data,
            quantile_points,
        )

        if apply_continuity_correction:
            segment_edf_at_quantiles -= 1 / (2 * segment_length)

        # Clip to within (0, 1) to avoid log(0) issues:
        segment_edf_at_quantiles = np.clip(segment_edf_at_quantiles, 1e-10, 1 - 1e-10)
        one_minus_segment_edf_at_quantiles = 1 - segment_edf_at_quantiles

        segment_ll_at_mle = (
            (-2.0 * integrated_edf_scaling / num_quantiles)
            * segment_length
            * (
                np.sum(
                    segment_edf_at_quantiles * np.log(segment_edf_at_quantiles)
                    + one_minus_segment_edf_at_quantiles
                    * np.log(one_minus_segment_edf_at_quantiles)
                )
            )
        )

        # The cost equals twice the negative integrated log-likelihood:
        segment_costs[i] = -2.0 * segment_ll_at_mle

    return segment_costs


@njit
def make_cumulative_edf_cache(
    xs: np.ndarray, quantile_points: np.ndarray
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
    quantile_points : np.ndarray
        The quantile values at which to compute the cumulative counts.

    Returns
    -------
    np.ndarray
        A 2D array where each row corresponds to a quantile value and contains the
        cumulative counts of values less than or equal to that quantile.
    """
    # Cache cumulative sums of the empirical distribution function (EDF):
    lte_quantile_point_mask = (xs[:, None] < quantile_points[None, :]).astype(
        np.float64
    )

    # Add 0.5 to the count for samples equal to the quantile values:
    lte_quantile_point_mask += 0.5 * (xs[:, None] == quantile_points[None, :])

    cumulative_edf_quantiles = col_cumsum(lte_quantile_point_mask, init_zero=True)

    return cumulative_edf_quantiles


@njit
def optimized_approximate_mle_edf_cost_cached_edf(
    cumulative_edf_quantiles: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    scratch_array: np.ndarray | None = None,
    segment_costs: np.ndarray | None = None,
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
    n_samples, num_quantiles = cumulative_edf_quantiles.shape

    # This is a placeholder for the actual implementation.
    # In practice, this function would compute the cost based on the
    # first num_quantiles terms.
    if num_quantiles <= 0:
        raise ValueError("num_quantiles must be a positive integer.")
    if num_quantiles > n_samples - 2:
        raise ValueError(
            "num_quantiles should not be greater than the number of samples minus 2."
        )

    if segment_costs is None:
        segment_costs = np.zeros(len(segment_starts), dtype=np.float64)

    if scratch_array is None:
        segment_edf_at_quantiles = np.zeros(num_quantiles, dtype=np.float64)
        one_minus_segment_edf_at_quantiles = np.zeros(num_quantiles, dtype=np.float64)
        log_segment_edf_at_quantiles = np.zeros(num_quantiles, dtype=np.float64)
    else:
        segment_edf_at_quantiles = scratch_array[0, :]
        one_minus_segment_edf_at_quantiles = scratch_array[1, :]
        log_segment_edf_at_quantiles = scratch_array[2, :]

    # Constant scaling term from the approximation of the integrated log-likelihood:
    edf_integration_scale = -np.log(2 * n_samples - 1)

    # Iterate over each segment defined by the starts and ends:
    for i, (segment_start, segment_end) in enumerate(zip(segment_starts, segment_ends)):
        segment_length = segment_end - segment_start
        if segment_length <= 0:
            raise ValueError("Invalid segment length.")

        # Shifted by 1 to account for the row of zeros at the start:
        segment_edf_at_quantiles[:] = cumulative_edf_quantiles[segment_end, :]
        segment_edf_at_quantiles[:] -= cumulative_edf_quantiles[segment_start, :]
        segment_edf_at_quantiles[:] /= float(segment_length)

        # Clip to within (0, 1) to avoid log(0) issues: Write to last argument
        np.clip(segment_edf_at_quantiles, 1e-10, 1 - 1e-10, segment_edf_at_quantiles)

        ### Begin computing integrated log-likelihood for the segment ###
        segment_ll_at_mle = 0.0

        # Compute the first term: sum(F(t))*log(F(t))
        np.log(segment_edf_at_quantiles, log_segment_edf_at_quantiles)

        # Multiply together, storing in one_minus_segment_edf_at_quantiles:
        np.multiply(
            segment_edf_at_quantiles,
            log_segment_edf_at_quantiles,
            one_minus_segment_edf_at_quantiles,
        )
        segment_ll_at_mle += np.sum(one_minus_segment_edf_at_quantiles)

        # Compute the second term: sum(1 - F(t))*log(1 - F(t))
        one_minus_segment_edf_at_quantiles[:] = 1 - segment_edf_at_quantiles
        np.log(one_minus_segment_edf_at_quantiles, log_segment_edf_at_quantiles)

        # Multiply together, storing in segment_edf_at_quantiles:
        np.multiply(
            one_minus_segment_edf_at_quantiles,
            log_segment_edf_at_quantiles,
            segment_edf_at_quantiles,
        )
        segment_ll_at_mle += np.sum(segment_edf_at_quantiles)

        segment_ll_at_mle *= (
            -2.0 * edf_integration_scale / num_quantiles
        ) * segment_length
        ### Done computing integrated log-likelihood for the segment ###

        # The cost equals twice the negative integrated log-likelihood:
        segment_costs[i] = (-2.0) * segment_ll_at_mle

    return segment_costs


@njit
def int_approximate_mle_edf_cost_cached_edf(
    cumulative_edf_quantiles: np.ndarray,
    binomial_ll_lookup: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    scratch_array: np.ndarray | None = None,
    segment_costs: np.ndarray | None = None,
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
    n_samples, num_quantiles = cumulative_edf_quantiles.shape
    lookup_size = len(binomial_ll_lookup)

    # This is a placeholder for the actual implementation.
    # In practice, this function would compute the cost based on the
    # first num_quantiles terms.
    if num_quantiles <= 0:
        raise ValueError("num_quantiles must be a positive integer.")
    if num_quantiles > n_samples - 2:
        raise ValueError(
            "num_quantiles should not be greater than the number of samples minus 2."
        )

    if segment_costs is None:
        segment_costs = np.zeros(len(segment_starts), dtype=np.float64)

    if scratch_array is None:
        segment_edf_at_quantiles = np.zeros(num_quantiles, dtype=np.int32)
    else:
        segment_edf_at_quantiles = scratch_array[0, :]

    # Constant scaling term from the approximation of the integrated log-likelihood:
    edf_integration_scale = -np.log(2 * n_samples - 1)

    # Iterate over each segment defined by the starts and ends:
    for i, (segment_start, segment_end) in enumerate(zip(segment_starts, segment_ends)):
        segment_length = segment_end - segment_start
        # if segment_length <= 0:
        #     raise ValueError("Invalid segment length.")

        # Shifted by 1 to account for the row of zeros at the start:
        segment_edf_at_quantiles[:] = cumulative_edf_quantiles[segment_end, :]
        segment_edf_at_quantiles[:] -= cumulative_edf_quantiles[segment_start, :]

        # Map [0, 1] to lookup indices [0, lookup_size - 1]:
        segment_edf_at_quantiles[:] *= lookup_size - 1

        # (Integer) Divide by segment length: (rounding down)
        segment_edf_at_quantiles[:] //= segment_length

        segment_ll_at_mle = np.sum(binomial_ll_lookup[segment_edf_at_quantiles])
        segment_ll_at_mle *= (
            -2.0 * edf_integration_scale / num_quantiles
        ) * segment_length

        # The cost equals twice the negative integrated log-likelihood:
        segment_costs[i] = (-2.0) * segment_ll_at_mle

    return segment_costs


class EmpiricalDistributionCost(BaseCost):
    """
    Empirical Distribution Cost.

    Computationally efficient approximate empirical distribution cost[1]_.
    Uses the integrated log-likelihood of the empirical distribution function (EDF)
    to evaluate the cost for each segment defined by the cuts, in a non-parametric way.

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

    References
    ----------
    .. [1] Haynes, K., Fearnhead, P. & Eckley, I.A. A computationally efficient
    nonparametric approach for changepoint detection. Stat Comput 27, 1293-1305 (2017).
    https://doi.org/10.1007/s11222-016-9687-5
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "distribution_type": "None",
        "is_conditional": False,
        "is_aggregated": False,
        "supports_fixed_param": True,
    }

    def __init__(
        self,
        param: tuple[ArrayLike, ArrayLike] | None = None,
        num_approximation_quantiles: int | None = None,
        use_cache: bool = True,
    ):
        # Initialize the base class
        super().__init__(param)

        # Store parameters:
        self.use_cache = use_cache
        self.num_approximation_quantiles = num_approximation_quantiles

        self.num_quantiles_ = None  # Will be set during fitting

        self._edf_cache = None  # Cache for empirical distribution function

        # TODO: Make x*log(x) + (1-x)*log(1-x) on [0.0, 1.0] a lookup table!
        self._binomial_ll_lookup_size = 1000
        binomial_ll_lookup_step_size = 1.0 / self._binomial_ll_lookup_size
        binomial_ll_lookup_points = np.linspace(
            binomial_ll_lookup_step_size,
            1.0 - binomial_ll_lookup_step_size,
            num=self._binomial_ll_lookup_size - 2,
            endpoint=True,
        )
        self._binomial_ll_lookup = np.zeros(
            self._binomial_ll_lookup_size, dtype=np.float64
        )
        self._binomial_ll_lookup[1:-1] = binomial_ll_lookup_points * np.log(
            binomial_ll_lookup_points
        ) + (1.0 - binomial_ll_lookup_points) * np.log(1.0 - binomial_ll_lookup_points)

        # Storage for fixed samples and quantiles:
        self.quantile_points_: np.ndarray | None = None
        self._quantile_weights: np.ndarray | None = None
        self.fixed_quantiles_: np.ndarray | None = None
        self._log_fixed_quantiles: np.ndarray | None = None
        self._one_minus_fixed_quantiles: np.ndarray | None = None
        self._log_one_minus_fixed_quantiles: np.ndarray | None = None

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
        n_samples, n_columns = X.shape

        if self._param is None:
            # Calculate number of quantiles to use if not provided:
            if self.num_approximation_quantiles is None:
                self.num_quantiles_ = int(np.ceil(4 * np.log(n_samples)))
            else:
                self.num_quantiles_ = self.num_approximation_quantiles

            self.quantile_points_ = make_approximate_mle_edf_cost_quantile_points(
                X, num_quantiles=self.num_quantiles_
            )

        else:
            self.quantile_points_, self.fixed_quantiles_ = self._param
            self.num_quantiles_ = self.fixed_quantiles_.shape[0]

            self._log_fixed_quantiles = np.log(self.fixed_quantiles_)
            self._log_one_minus_fixed_quantiles = np.log(1.0 - self.fixed_quantiles_)

            # Store the quantile integration weights for each data column:
            self._quantile_weights = np.zeros(self.fixed_quantiles_.shape)
            for col in range(n_columns):
                self._quantile_weights[:, col] = make_fixed_cdf_cost_quantile_weights(
                    self.fixed_quantiles_[:, col],
                    self.quantile_points_[:, col],
                )

        if self.use_cache:
            self._edf_cache = [
                make_cumulative_edf_cache(
                    self._X[:, col], quantile_points=self.quantile_points_[:, col]
                )
                for col in range(n_columns)
            ]
            self._int_edf_cache = [
                edf_cache.astype(np.int32) for edf_cache in self._edf_cache
            ]

        # Memory used during the evaluation of the cost:
        self._scratch_array = np.zeros(
            (3, self.num_quantiles_),
            dtype=np.float64,
        )

        self._int_scratch_array = np.zeros(
            (3, self.num_quantiles_),
            dtype=np.int32,
        )

        return self

    def _check_fixed_param(self, param: tuple[np.ndarray, np.ndarray], X: np.ndarray):
        fixed_samples, fixed_cdf_quantiles = param

        if not (fixed_samples.shape == fixed_cdf_quantiles.shape):
            raise ValueError("All samples must have a corresponding fixed quantile.")

        # Standardize the shapes of fixed samples and quantiles:
        if fixed_samples.ndim == 1:
            fixed_samples = fixed_samples.reshape(-1, 1)
        elif fixed_cdf_quantiles.ndim > 2:
            raise ValueError(
                "Fixed samples must be 1D or 2D arrays, "
                f"but got shape {fixed_samples.shape}."
            )
        if fixed_cdf_quantiles.ndim == 1:
            fixed_cdf_quantiles = fixed_cdf_quantiles.reshape(-1, 1)
        elif fixed_cdf_quantiles.ndim > 2:
            raise ValueError(
                "Fixed quantiles must be 1D or 2D arrays, "
                f"but got shape {fixed_cdf_quantiles.shape}."
            )

        if not np.all(np.diff(fixed_samples, axis=0) > 0):
            raise ValueError("Fixed samples must be sorted, and strictly increasing.")

        if not np.all(np.diff(fixed_cdf_quantiles, axis=0) > 0):
            raise ValueError(
                "Fixed CDF quantiles must be sorted, and strictly increasing."
            )

        if not (
            np.all(0 <= fixed_cdf_quantiles) and np.all(fixed_cdf_quantiles <= 1.0)
        ):
            raise ValueError(
                "Fixed quantiles must be within the closed interval [0, 1]"
            )

        # Clip fixed quantiles to avoid log(0) issues:
        fixed_cdf_quantiles = np.clip(fixed_cdf_quantiles, 1.0e-10, 1.0 - 1.0e-10)

        # Repeat the fixed samples and quantiles to match the number of columns in X:
        if fixed_samples.shape[1] == 1 and X.shape[1] > 1:
            fixed_samples = np.tile(fixed_samples, (1, X.shape[1]))
            fixed_cdf_quantiles = np.tile(fixed_cdf_quantiles, (1, X.shape[1]))

        return fixed_samples, fixed_cdf_quantiles

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
        n_cols = self._X.shape[1]

        costs = np.zeros((n_intervals, n_cols))
        for col in range(n_cols):
            if self.use_cache:
                # optimized_approximate_mle_edf_cost_cached_edf(
                #     self._edf_cache[col],
                #     segment_starts=starts,
                #     segment_ends=ends,
                #     scratch_array=self._scratch_array,
                #     segment_costs=costs[:, col],
                # )
                int_approximate_mle_edf_cost_cached_edf(
                    self._int_edf_cache[col],
                    self._binomial_ll_lookup,
                    segment_starts=starts,
                    segment_ends=ends,
                    scratch_array=self._int_scratch_array,
                    segment_costs=costs[:, col],
                )
            else:
                approximate_mle_edf_cost(
                    self._X[:, col],
                    quantile_points=self.quantile_points_[:, col],
                    segment_starts=starts,
                    segment_ends=ends,
                    segment_costs=costs[:, col],
                )

        return costs

    def _evaluate_fixed_param(self, starts: np.ndarray, ends: np.ndarray):
        n_intervals = len(starts)
        n_cols = self._X.shape[1]

        costs = np.zeros((n_intervals, n_cols))
        for col in range(n_cols):
            costs[:, col] = fixed_cdf_empirical_distribution_cost_cached_edf(
                cumulative_edf_quantiles=self._edf_cache[col],
                segment_starts=starts,
                segment_ends=ends,
                log_fixed_quantiles=self._log_fixed_quantiles[:, col],
                log_one_minus_fixed_quantiles=self._log_one_minus_fixed_quantiles[
                    :, col
                ],
                quantile_weights=self._quantile_weights[:, col],
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
        if self.is_fitted and self.num_quantiles_ is not None:
            return self.num_quantiles_ + 2
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
        fixed_samples = np.array(
            [
                -1.95996398,
                -1.64485363,
                -1.03643339,
                -0.52440051,
                0.0,
                0.52440051,
                1.03643339,
                1.64485363,
                1.95996398,
            ]
        )
        fixed_quantiles = np.array(
            [0.025, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 0.975]
        )
        # Define two different parameter sets for testing
        params1 = {"param": None, "use_cache": True, "num_approximation_quantiles": 10}
        params2 = {
            "param": (fixed_samples, fixed_quantiles),
            "use_cache": True,
            "num_approximation_quantiles": None,
        }
        return [params1, params2]
