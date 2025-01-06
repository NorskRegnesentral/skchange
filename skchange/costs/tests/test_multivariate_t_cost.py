from time import perf_counter

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.stats as st
from scipy.special import digamma, gammaln, polygamma

from skchange.costs.multivariate_t_cost import (
    _multivariate_t_log_likelihood,
    isotropic_t_dof_estimate,
    iterative_mv_t_dof_estimate,
    kurtosis_t_dof_estimate,
    loo_iterative_mv_t_dof_estimate,
    maximum_likelihood_mv_t_scale_matrix,
)
from skchange.utils.numba import numba_available


def estimate_mv_t_dof_nojit(
    centered_samples: np.ndarray, zero_norm_tol=1.0e-6
) -> float:
    """Estimate the degrees of freedom of a multivariate t-distribution.

    From: A Novel Parameter Estimation Algorithm for the Multivariate
          t-Distribution and Its Application to Computer Vision.
    """
    sample_dim = centered_samples.shape[1]

    squared_norms = np.sum(centered_samples * centered_samples, axis=1)

    log_norm_sq: np.ndarray
    log_norm_sq = np.log(squared_norms[squared_norms > zero_norm_tol**2])
    log_norm_sq_var = log_norm_sq.var(ddof=0)

    b = log_norm_sq_var - polygamma(1, sample_dim / 2.0)
    t_dof_estimate = (1 + np.sqrt(1 + 4 * b)) / b

    return t_dof_estimate


def estimate_scale_matrix_trace_nojit(centered_samples: np.ndarray, dof: float):
    """Estimate the scale parameter of the MLE covariance matrix."""
    p = centered_samples.shape[1]
    squared_norms = np.sum(centered_samples * centered_samples, axis=1)
    z_bar = np.log(squared_norms[squared_norms > 1.0e-12]).mean()
    log_alpha = z_bar - np.log(dof) + digamma(0.5 * dof) - digamma(p / 2.0)
    return p * np.exp(log_alpha)


def initial_scale_matrix_estimate_nojit(
    centered_samples: np.ndarray,
    t_dof: float,
    num_zeroed_samples: int = 0,
    apply_trace_correction: bool = True,
):
    """Estimate the scale matrix given centered samples and degrees of freedom."""
    n, p = centered_samples.shape
    num_effective_samples = n - num_zeroed_samples

    sample_covariance_matrix = (
        centered_samples.T @ centered_samples
    ) / num_effective_samples

    if apply_trace_correction:
        scale_trace_estimate = estimate_scale_matrix_trace_nojit(
            centered_samples, t_dof
        )
        sample_covariance_matrix *= scale_trace_estimate / np.trace(
            sample_covariance_matrix
        )

    return sample_covariance_matrix


def scale_matrix_fixed_point_iteration(
    scale_matrix: np.ndarray,
    t_dof: float,
    centered_samples: np.ndarray,
    num_zeroed_samples: int = 0,
):
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape

    # Subtract the number of 'zeroed' samples:
    effective_num_samples = n - num_zeroed_samples

    inv_cov_2d = sla.solve(
        scale_matrix, np.eye(p), assume_a="pos", overwrite_a=False, overwrite_b=True
    )
    z_scores = np.einsum("ij,jk,ik->i", centered_samples, inv_cov_2d, centered_samples)

    sample_weight = (p + t_dof) / (t_dof + z_scores)
    weighted_samples = centered_samples * sample_weight[:, np.newaxis]

    reconstructed_scale_matrix = (
        weighted_samples.T @ centered_samples
    ) / effective_num_samples

    return reconstructed_scale_matrix


def solve_mle_scale_matrix_nojit(
    initial_scale_matrix: np.ndarray,
    centered_samples: np.ndarray,
    t_dof: float,
    num_zeroed_samples: int = 0,
    max_iter: int = 50,
    reverse_tol: float = 1.0e-3,
) -> np.ndarray:
    """Perform fixed point iterations for the MLE scale matrix."""
    scale_matrix = initial_scale_matrix.copy()
    temp_cov_matrix = initial_scale_matrix.copy()

    # Compute the MLE covariance matrix using fixed point iteration:
    for iteration in range(max_iter):
        temp_cov_matrix = scale_matrix_fixed_point_iteration(
            scale_matrix=scale_matrix,
            t_dof=t_dof,
            centered_samples=centered_samples,
            num_zeroed_samples=num_zeroed_samples,
        )

        # Note: 'ord = None' computes the Frobenius norm.
        residual = np.linalg.norm(temp_cov_matrix - scale_matrix, ord=None)

        scale_matrix = temp_cov_matrix.copy()
        if residual < reverse_tol:
            break

    return scale_matrix, iteration


def maximum_likelihood_scale_matrix_nojit(
    centered_samples: np.ndarray,
    t_dof: float,
    reverse_tol: float = 1.0e-3,
    max_iter: int = 50,
    num_zeroed_samples: int = 0,
    initial_trace_correction: bool = True,
) -> np.ndarray:
    """Compute the MLE scale matrix for a multivariate t-distribution.

    Parameters
    ----------
    centered_samples : np.ndarray
        The centered samples from the multivariate t-distribution.
    dof : float
        The degrees of freedom of the multivariate t-distribution.

    Returns
    -------
    np.ndarray
        The MLE covariance matrix of the multivariate t-distribution.
    """
    # Initialize the scale matrix:
    mle_scale_matrix = initial_scale_matrix_estimate_nojit(
        centered_samples,
        t_dof,
        num_zeroed_samples=num_zeroed_samples,
        apply_trace_correction=initial_trace_correction,
    )

    mle_scale_matrix, inner_iterations = solve_mle_scale_matrix_nojit(
        initial_scale_matrix=mle_scale_matrix,
        centered_samples=centered_samples,
        t_dof=t_dof,
        num_zeroed_samples=num_zeroed_samples,
        max_iter=max_iter,
        reverse_tol=reverse_tol,
    )

    return mle_scale_matrix


def approximate_mv_t_scale_matrix_gradient(
    scale_matrix: np.ndarray,
    centered_samples: np.ndarray,
    t_dof: float,
    epsilon: float = 1e-4,
):
    """
    Approximate the gradient of the scale matrix.

    Parameters
    ----------
    centered_samples : `np.ndarray`
        Centered samples.
    scale_matrix : `np.ndarray`
        Scale matrix.
    epsilon : `float`, optional (default=1e-4)
        Epsilon.

    Returns
    -------
    grad : `np.ndarray`
        Gradient.
    """
    p = centered_samples.shape[1]
    grad = np.zeros((p, p))

    for i in range(p):
        for j in range(p):
            scale_matrix_plus = scale_matrix.copy()
            scale_matrix_plus[i, j] += epsilon

            scale_matrix_minus = scale_matrix.copy()
            scale_matrix_minus[i, j] -= epsilon

            ll_plus = st.multivariate_t.logpdf(
                centered_samples, loc=np.zeros(p), shape=scale_matrix_plus, df=t_dof
            ).sum()
            ll_minus = st.multivariate_t.logpdf(
                centered_samples, loc=np.zeros(p), shape=scale_matrix_minus, df=t_dof
            ).sum()

            grad[i, j] = (ll_plus - ll_minus) / (2 * epsilon)

    return grad


def _multivariate_t_log_likelihood_like_scipy(
    scale_matrix, centered_samples: np.ndarray, t_dof: float
) -> float:
    """Calculate the log likelihood of a multivariate t-distribution.

    Directly from the definition of the multivariate t-distribution.
    Implementation taken from the scipy implementation of the multivariate t-distribution.
    Source: https://en.wikipedia.org/wiki/Multivariate_t-distribution

    Parameters
    ----------
    scale_matrix : np.ndarray
        The scale matrix of the multivariate t-distribution.
    centered_samples : np.ndarray
        The centered samples from the multivariate t-distribution.
    t_dof : float
        The degrees of freedom of the multivariate t-distribution.

    Returns
    -------
    float
        The log likelihood of the multivariate t-distribution.
    """
    num_samples, sample_dim = centered_samples.shape

    # Compute the log likelihood of the segment using numba, but similar to scipy:
    scale_spectrum, scale_eigvecs = np.linalg.eigh(scale_matrix)
    log_det_scale_matrix = np.sum(np.log(scale_spectrum))

    scale_sqrt_inv = np.multiply(scale_eigvecs, 1.0 / np.sqrt(scale_spectrum))
    mahalonobis_distances = np.square(np.dot(centered_samples, scale_sqrt_inv)).sum(
        axis=-1
    )

    # Normalization constants:
    exponent = 0.5 * (t_dof + sample_dim)
    A = gammaln(exponent)
    B = gammaln(0.5 * t_dof)
    C = 0.5 * sample_dim * np.log(t_dof * np.pi)
    D = 0.5 * log_det_scale_matrix
    normalization_contribution = num_samples * (A - B - C - D)

    sample_contributions = -exponent * np.log1p(mahalonobis_distances / t_dof)

    total_log_likelihood = normalization_contribution + sample_contributions.sum()

    return total_log_likelihood


def test_mv_t_log_likelihood(seed=4125, num_samples=100, p=8, t_dof=5.0):
    # TODO: Parametrize and test over wide range of input values.
    np.random.seed(seed)

    mv_t_samples = st.multivariate_t(loc=np.zeros(p), shape=np.eye(p), df=t_dof).rvs(
        num_samples
    )

    sample_medians = np.median(mv_t_samples, axis=0)
    X_centered = mv_t_samples - sample_medians
    mle_scale_matrix = maximum_likelihood_mv_t_scale_matrix(X_centered, t_dof)

    ll_scipy = (
        st.multivariate_t(loc=sample_medians, shape=mle_scale_matrix, df=t_dof)
        .logpdf(mv_t_samples)
        .sum()
    )

    ll_manual_scipy = _multivariate_t_log_likelihood_like_scipy(
        scale_matrix=mle_scale_matrix, centered_samples=X_centered, t_dof=t_dof
    )

    ll_simple = _multivariate_t_log_likelihood(
        scale_matrix=mle_scale_matrix, centered_samples=X_centered, dof=t_dof
    )
    ll_differences = np.diff(np.array([ll_scipy, ll_manual_scipy, ll_simple, ll_scipy]))

    np.testing.assert_allclose(ll_differences, 0, atol=1e-4)


def test_scale_matrix_mle(seed=4125):
    """Test scale matrix MLE."""
    # TODO: Parametrize and test over wide range of input values.
    np.random.seed(seed)
    n_samples = 50
    p = 3
    t_dof = 5.0

    random_nudge = np.random.randn(p).reshape(-1, 1)
    true_scale_matrix = np.eye(p) + 0.5 * random_nudge @ random_nudge.T

    true_mean = np.arange(p) * (-1 * np.ones(p)).cumprod()

    mv_t_samples = st.multivariate_t(
        loc=true_mean, shape=true_scale_matrix, df=t_dof
    ).rvs(n_samples)

    sample_medians = np.median(mv_t_samples, axis=0)
    centered_samples = mv_t_samples - sample_medians

    # Estimate the scale matrix:
    mle_scale_matrix = maximum_likelihood_scale_matrix_nojit(
        centered_samples, t_dof, reverse_tol=1.0e-6
    )

    # Compute approximate gradients:
    mle_scale_matrix_grad = approximate_mv_t_scale_matrix_gradient(
        mle_scale_matrix, centered_samples, t_dof
    )
    true_scale_matrix_grad = approximate_mv_t_scale_matrix_gradient(
        true_scale_matrix, centered_samples, t_dof
    )

    # Assure that the MLE scale matrix gradient is close to zero:
    np.testing.assert_allclose(mle_scale_matrix_grad, 0.0, atol=1e-5)

    # Assure that the norm of the gradient for the true scale matrix is larger:
    assert np.linalg.norm(true_scale_matrix_grad) > np.linalg.norm(
        mle_scale_matrix_grad
    ), "True scale matrix gradient is not larger than the MLE gradient."

    # Assure that we've increased the log-likelihood with the MLE scale matrix:
    true_scale_matrix_ll = st.multivariate_t.logpdf(
        centered_samples, loc=np.zeros(p), shape=true_scale_matrix, df=t_dof
    ).sum()

    mle_scale_matrix_ll = st.multivariate_t.logpdf(
        centered_samples, loc=np.zeros(p), shape=mle_scale_matrix, df=t_dof
    ).sum()

    assert (
        mle_scale_matrix_ll > true_scale_matrix_ll
    ), "MLE log-likelihood is not maximal."


def test_scale_matrix_numba_benchmark(
    n_trials=10,
    n_samples=1_000,
    p=3,
    t_dof=5.0,
    initial_trace_correction=True,
    verbose=False,
):
    """Benchmark numba vs non-numba scale matrix computation."""
    if not numba_available:
        pytest.skip("Numba not available, cannot test benchmark.")

    times_njit = []
    times_normal = []

    for seed in range(n_trials):
        np.random.seed(seed)

        # Generate test data
        random_nudge = np.random.randn(p).reshape(-1, 1)
        true_scale_matrix = np.eye(p) + 0.5 * random_nudge @ random_nudge.T
        true_mean = np.arange(p) * (-1 * np.ones(p)).cumprod()
        mv_t_samples = st.multivariate_t(
            loc=true_mean, shape=true_scale_matrix, df=t_dof
        ).rvs(n_samples)
        centered_samples = mv_t_samples - np.median(mv_t_samples, axis=0)

        if seed == 0:
            # Ensure compilation time is not measured:
            maximum_likelihood_mv_t_scale_matrix(
                centered_samples,
                t_dof,
                initial_trace_correction=initial_trace_correction,
            )

        # Time numba version
        start = perf_counter()
        numba_mle_scale_matrix = maximum_likelihood_mv_t_scale_matrix(
            centered_samples, t_dof, initial_trace_correction=initial_trace_correction
        )
        end = perf_counter()
        times_njit.append(end - start)

        # Time normal version
        start = perf_counter()
        normal_mle_scale_matrix = maximum_likelihood_scale_matrix_nojit(
            centered_samples, t_dof, initial_trace_correction=initial_trace_correction
        )
        end = perf_counter()
        times_normal.append(end - start)

        # Assert numba version is correct:
        np.testing.assert_allclose(
            numba_mle_scale_matrix, normal_mle_scale_matrix, atol=1e-10
        )

    # Assert numba version is faster on average:
    mean_numba_time = np.mean(times_njit)
    mean_normal_time = np.mean(times_normal)
    numba_speedup = mean_normal_time / mean_numba_time

    if verbose:
        print(f"Mean time normal: {mean_normal_time:.3e}")
        print(f"Mean time numba: {mean_numba_time:.3e}")
        print(f"Numba speedup: {numba_speedup:.3f}")

    assert numba_speedup > 1, "Numba version should be faster"


def test_isotropic_and_kurtosis_t_dof_estimates():
    seed = 4125
    n_samples = 1000
    p = 5
    t_dof = 5.0

    np.random.seed(seed)

    # Spd covariance matrix:
    random_nudge = np.random.randn(p).reshape(-1, 1)
    cov = np.eye(p) + 0.5 * random_nudge @ random_nudge.T

    mean = np.arange(p) * (-1 * np.ones(p)).cumprod()

    mv_t_dist = st.multivariate_t(loc=mean, shape=cov, df=t_dof)
    mv_t_samples = mv_t_dist.rvs(n_samples)

    sample_medians = np.median(mv_t_samples, axis=0)
    centered_samples = mv_t_samples - sample_medians

    # Test isotropic estimate:
    isotropic_dof = isotropic_t_dof_estimate(centered_samples)
    assert isotropic_dof > 0, "Isotropic dof estimate should be positive."
    assert np.abs(isotropic_dof - t_dof) < 1.0, "Isotropic dof estimate is off."

    # Test kurtosis estimate:
    kurtosis_dof = kurtosis_t_dof_estimate(centered_samples)
    assert kurtosis_dof > 0, "Kurtosis dof estimate should be positive."
    assert np.abs(kurtosis_dof - t_dof) < 1.0, "Kurtosis dof estimate is off."


def test_iterative_t_dof_estimate():
    seed = 4125
    n_samples = 150
    p = 5
    t_dof = 5.0

    np.random.seed(seed)

    # Spd covariance matrix:
    random_nudge = np.random.randn(p).reshape(-1, 1)
    cov = np.eye(p) + 0.5 * random_nudge @ random_nudge.T

    mean = np.arange(p) * (-1 * np.ones(p)).cumprod()

    mv_t_dist = st.multivariate_t(loc=mean, shape=cov, df=t_dof)
    mv_t_samples = mv_t_dist.rvs(n_samples)

    sample_medians = np.median(mv_t_samples, axis=0)
    centered_samples = mv_t_samples - sample_medians

    # Test data-driven estimate:
    initial_t_dof_estimate = 10.0
    iterative_dof_estimate = iterative_mv_t_dof_estimate(
        centered_samples, initial_dof=initial_t_dof_estimate
    )
    assert iterative_dof_estimate > 0, "Data-driven dof estimate should be positive."
    assert (
        np.abs(iterative_dof_estimate - t_dof) < 1.0
    ), "Data-driven dof estimate is off."


def test_loo_iterative_t_dof_estimate():
    seed = 4125
    n_samples = 150
    p = 5
    t_dof = 5.0

    np.random.seed(seed)

    # Spd covariance matrix:
    random_nudge = np.random.randn(p).reshape(-1, 1)
    cov = np.eye(p) + 0.5 * random_nudge @ random_nudge.T

    mean = np.arange(p) * (-1 * np.ones(p)).cumprod()

    mv_t_dist = st.multivariate_t(loc=mean, shape=cov, df=t_dof)
    mv_t_samples = mv_t_dist.rvs(n_samples)

    sample_medians = np.median(mv_t_samples, axis=0)
    centered_samples = mv_t_samples - sample_medians

    # Test data-driven estimate:
    initial_t_dof_estimate = 10.0
    loo_iterative_dof = loo_iterative_mv_t_dof_estimate(
        centered_samples, initial_dof=initial_t_dof_estimate
    )
    assert loo_iterative_dof > 0, "LOO data-driven dof estimate should be positive."
    assert (
        np.abs(loo_iterative_dof - t_dof) < 0.15
    ), "LOO data-driven dof estimate is off."


def test_iterative_dof_estimate_returns_inf_on_gaussian_data():
    seed = 4125
    n_samples = 150
    p = 5

    np.random.seed(seed)

    mean = np.arange(p) * (-1 * np.ones(p)).cumprod()
    cov = np.eye(p)

    mv_normal_samples = st.multivariate_normal(mean=mean, cov=cov).rvs(n_samples)

    sample_medians = np.median(mv_normal_samples, axis=0)
    centered_samples = mv_normal_samples - sample_medians

    # Test data-driven estimate:
    initial_t_dof_estimate = 4.0
    iterative_dof_estimate = iterative_mv_t_dof_estimate(
        centered_samples,
        initial_dof=initial_t_dof_estimate,
        infinite_dof_threshold=50.0,
        max_iter=100,
    )

    assert np.isposinf(
        iterative_dof_estimate
    ), "Dof estimate should be infinite on Gaussian data."


def test_loo_iterative_dof_estimate_returns_inf_on_gaussian_data():
    seed = 4125
    n_samples = 500
    p = 5

    np.random.seed(seed)

    mean = np.arange(p) * (-1 * np.ones(p)).cumprod()
    cov = np.eye(p)

    mv_normal_samples = st.multivariate_normal(mean=mean, cov=cov).rvs(n_samples)

    sample_medians = np.median(mv_normal_samples, axis=0)
    centered_samples = mv_normal_samples - sample_medians

    # Test data-driven estimate:
    initial_t_dof_estimate = 4.0
    loo_iterative_dof = loo_iterative_mv_t_dof_estimate(
        centered_samples,
        initial_dof=initial_t_dof_estimate,
        infinite_dof_threshold=50.0,
        max_iter=100,
    )

    assert np.isposinf(
        loo_iterative_dof
    ), "Dof estimate should be infinite on Gaussian data."
