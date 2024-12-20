"""Example of fitting a multivariate t-distribution to data."""

# %%
import timeit
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpy.linalg as la
import pymanopt as pm
import scipy.linalg as sla
import scipy.stats as st
from jax import jacfwd
from scipy.linalg import solve_continuous_lyapunov
from scipy.special import digamma, polygamma


# %%
def spd_Bures_Wasserstein_exponential(p: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute the Bures-Wasserstein exponential map on the SPD manifold.

    Parameters
    ----------
    p : np.ndarray
        The base point on the SPD manifold. Must be symmetric positive definite.
    X : np.ndarray
        The tangent vector at the base point. Must be symmetric.

    Returns
    -------
    np.ndarray
        The exponential map of the tangent vector at the base point.
    """
    result_q = p.copy()
    result_q += X

    lyapunov_sol = solve_continuous_lyapunov(p, X)
    result_q += lyapunov_sol @ p @ lyapunov_sol

    return result_q


def estimate_mle_cov_scale(centered_samples: np.ndarray, dof: float):
    """Estimate the scale parameter of the MLE covariance matrix."""
    p = centered_samples.shape[1]
    z_bar = np.log(np.sum(centered_samples * centered_samples, axis=1)).mean()
    log_alpha = z_bar - np.log(dof) + digamma(0.5 * dof) - digamma(p / 2.0)
    return np.exp(log_alpha)


@jax.jit
def mle_cov_residual_chol_solve_jax(
    cov_2d: np.ndarray, dof: float, centered_samples: np.ndarray
) -> np.ndarray:
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape

    inv_cov_2d = jsp.linalg.solve(cov_2d, jnp.eye(p), assume_a="pos")
    z_scores = jnp.einsum("ij,jk,ik->i", centered_samples, inv_cov_2d, centered_samples)

    scales = (p + dof) / (dof + z_scores)
    weighted_samples = centered_samples * scales[:, jnp.newaxis]
    reconstructed_cov = weighted_samples.T @ centered_samples / n

    return cov_2d - reconstructed_cov


@jax.jit
def mle_1d_cov_residual_chol_solve_jax(
    cov_1d: np.ndarray, dof: float, centered_samples: np.ndarray
) -> np.ndarray:
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape
    cov_2d = cov_1d.reshape(p, p)

    inv_cov_2d = jsp.linalg.solve(cov_2d, jnp.eye(p), assume_a="pos")

    # Compute 'n' quadratic forms: (n, p) @ (p, p) @ (p, n) -> (n,)
    z_scores = jnp.einsum(
        "ij,jk,ki->i", centered_samples, inv_cov_2d, centered_samples.T
    )

    scales = (p + dof) / (dof + z_scores)
    weighted_samples = centered_samples * scales[:, jnp.newaxis]
    reconstructed_cov = weighted_samples.T @ centered_samples / n

    cov_diff = cov_2d - reconstructed_cov
    return cov_diff.reshape(-1)


@jax.jit
def mle_1d_cov_residual_jax(
    cov_1d: np.ndarray, dof: float, centered_samples: np.ndarray
):
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape
    cov_2d = cov_1d.reshape(p, p)
    inv_cov = jnp.linalg.inv(cov_2d)
    z_scores = jnp.sum(jnp.dot(centered_samples, inv_cov) * centered_samples, axis=1)
    scales = (p + dof) / (dof + z_scores)
    reconstructed_cov = (centered_samples.T * scales) @ centered_samples / n
    cov_diff = cov_2d - reconstructed_cov
    return cov_diff.reshape(-1)


# Compute the Jacobian of the MLE covariance residual function: (p * p,) -> (p *p,)
mle_1d_cov_residual = mle_1d_cov_residual_chol_solve_jax
# mle_1d_cov_residual = mle_1d_cov_residual_jax
jac_mle_1d_cov_residual = jacfwd(mle_1d_cov_residual, argnums=0)


def mv_t_newton_mle_covariance_matrix(
    centered_samples: np.ndarray,
    t_dof: float,
    retraction: Callable = spd_Bures_Wasserstein_exponential,
    reverse_tol: float = 1.0e-3,
    max_iter: int = 20,
) -> np.ndarray:
    """Compute the MLE covariance matrix for a multivariate t-distribution.

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
    n, p = centered_samples.shape

    covariance_2d: np.ndarray
    covariance_2d = centered_samples.T @ centered_samples / n

    alpha_estimate = estimate_mle_cov_scale(centered_samples, t_dof)
    contraction_estimate = alpha_estimate * p / np.trace(sample_covariance_matrix)
    covariance_2d *= contraction_estimate

    for i in range(max_iter):
        # Compute residual and its norm
        flat_residual = mle_1d_cov_residual(
            covariance_2d.reshape(-1), t_dof, centered_samples
        )
        residual_norm = la.norm(flat_residual)
        # print(f"Iteration {i}, residual norm: {residual_norm}")
        if residual_norm < reverse_tol:
            break

        flat_cov_jacobian = jac_mle_1d_cov_residual(
            covariance_2d.reshape(-1), t_dof, centered_samples
        )

        # Solve for Newton  step:
        newton_step = la.solve(flat_cov_jacobian, flat_residual)

        # Retract the step back to the manifold:
        tangent_matrix = newton_step.reshape(p, p)
        tangent_matrix = 0.5 * (tangent_matrix + tangent_matrix.T)

        new_cov = retraction(covariance_2d, -tangent_matrix)
        covariance_2d = new_cov

    return covariance_2d, i


# %%
@jax.jit
def mle_cov_fixed_point_jax(
    cov_2d: np.ndarray, dof: float, centered_samples: np.ndarray
):
    """Compute the MLE covariance residual for a mv_t distribution."""
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape

    inv_cov_2d = jsp.linalg.solve(cov_2d, jnp.eye(p), assume_a="pos")
    z_scores = jnp.einsum("ij,jk,ik->i", centered_samples, inv_cov_2d, centered_samples)

    scales = (p + dof) / (dof + z_scores)
    weighted_samples = centered_samples * scales[:, jnp.newaxis]
    reconstructed_cov = weighted_samples.T @ centered_samples / n

    return reconstructed_cov


# Function to compute the MLE covariance matrix for a multivariate t-distribution,
# using fixed point iteration:
def mv_t_fixed_point_mle_covariance_matrix(
    centered_samples: np.ndarray,
    dof: float,
    reverse_tol: float = 1.0e-3,
    max_iter: int = 20,
) -> np.ndarray:
    """Compute the MLE covariance matrix for a multivariate t-distribution.

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
    n, p = centered_samples.shape

    # Initialize the covariance matrix:
    covariance_matrix = centered_samples.T @ centered_samples / n
    alpha_estimate = estimate_mle_cov_scale(centered_samples, t_dof)
    contraction_estimate = alpha_estimate * p / np.trace(sample_covariance_matrix)
    covariance_matrix *= contraction_estimate

    temp_cov_matrix = covariance_matrix.copy()

    # Compute the MLE covariance matrix using fixed point iteration:
    for iteration in range(max_iter):
        temp_cov_matrix = mle_cov_fixed_point_jax(
            covariance_matrix, dof, centered_samples
        )
        residual = la.norm(temp_cov_matrix - covariance_matrix, ord="fro")
        covariance_matrix = temp_cov_matrix.copy()
        if residual < reverse_tol:
            break

        covariance_matrix = temp_cov_matrix.copy()
        if residual < reverse_tol:
            break

    return covariance_matrix, iteration


# %% Estimate the degrees of freedom of the multivariate t-distribution:
def estimate_mv_t_dof(centered_samples: np.ndarray) -> float:
    """Estimate the degrees of freedom of a multivariate t-distribution."""
    p = centered_samples.shape[1]

    log_norm_sq = np.log(np.sum(centered_samples * centered_samples, axis=1))
    log_norm_sq_var = log_norm_sq.var(ddof=0)

    b = log_norm_sq_var - polygamma(1, p / 2.0)
    t_dof = (1 + np.sqrt(1 + 4 * b)) / b

    return t_dof


# %%
def ellipitical_kurtosis_from_samples(centered_samples: np.ndarray) -> float:
    """Compute the kurtosis of a set of samples.

    From:
    Shrinking the eigenvalues of M-estimators of covariance matrix,
    subsection IV-A.
    by: Esa Ollila, Daniel P. Palomar, and Frédéric Pascal
    """
    kurtosis_per_dim = st.kurtosis(centered_samples, axis=0)
    averaged_kurtosis = kurtosis_per_dim.mean()
    return averaged_kurtosis / 3.0


def kurtosis_t_dof_estimate(centered_samples: np.ndarray) -> float:
    """Estimate the degrees of freedom of a multivariate t-distribution."""
    ellipitical_kurtosis = ellipitical_kurtosis_from_samples(centered_samples)
    t_dof = 2.0 / max(1.0e-3, ellipitical_kurtosis) + 4.0
    return t_dof


# %% Generate some data from a multivariate t-distribution:
# np.random.seed(42)
np.random.seed(40)

n_samples = 1000
p = 3
t_dof = 12.0

# Spd covariance matrix:
random_nudge = np.random.randn(p).reshape(-1, 1)
cov = np.eye(p) + 0.5 * random_nudge @ random_nudge.T
# la.cholesky(cov)
# la.eigvalsh(cov)

mean = np.arange(p) * (-1 * np.ones(p)).cumprod()

mv_t_dist = st.multivariate_t(loc=mean, shape=cov, df=t_dof)
mv_t_samples = mv_t_dist.rvs(n_samples)

sample_medians = np.median(mv_t_samples, axis=0)
centered_samples = mv_t_samples - sample_medians

sample_covariance_matrix = centered_samples.T @ centered_samples / n_samples

estimated_t_dof = estimate_mv_t_dof(centered_samples)
kurtosis_t_dof = kurtosis_t_dof_estimate(centered_samples)
avg_t_dof = (estimated_t_dof + kurtosis_t_dof) / 2.0
geo_mean_t_dof = np.sqrt(estimated_t_dof * kurtosis_t_dof)

print(f"True degrees of freedom: {t_dof}")
print(f"(visual) Estimated degrees of freedom: {estimated_t_dof}")
print(f"Kurtosis-based estimate: {kurtosis_t_dof}")
print(f"Average estimate: {avg_t_dof}")

# Cheap and good compromise between the two estimates:
# One estimate is biased up for low dof (kurtosis),
# and the other is biased down at high dof (visual).
print(f"Geometric mean estimate: {geo_mean_t_dof}")


# %%
def mv_t_samples_log_likelihood(
    samples: np.ndarray, mean: np.ndarray, cov: np.ndarray, dof: float
):
    mv_t_dist = st.multivariate_t(loc=mean, shape=cov, df=dof)
    return mv_t_dist.logpdf(samples).sum()


true_param_ll = mv_t_samples_log_likelihood(mv_t_samples, mean, cov, t_dof)
true_cov_ll = mv_t_samples_log_likelihood(mv_t_samples, sample_medians, cov, t_dof)

# Compute the MLE covariance matrix for the multivariate t-distribution:
fixed_point_mle_cov_matrix, num_fixed_point_iters = (
    mv_t_fixed_point_mle_covariance_matrix(
        centered_samples, t_dof, reverse_tol=1.0e-6, max_iter=50
    )
)

spd_manifold = pm.manifolds.positive_definite.SymmetricPositiveDefinite(n=p)
newton_mle_cov_matrix, num_newton_iters = mv_t_newton_mle_covariance_matrix(
    centered_samples,
    t_dof,
    retraction=spd_Bures_Wasserstein_exponential,
    # retraction=spd_manifold.retraction,
    reverse_tol=1.0e-6,
    max_iter=50,
)

alpha_estimate = estimate_mle_cov_scale(centered_samples, t_dof)
contraction_estimate = alpha_estimate * p / np.trace(sample_covariance_matrix)
initial_covariance_guess = contraction_estimate * sample_covariance_matrix

initial_coviariance_mle_ll = mv_t_samples_log_likelihood(
    mv_t_samples, sample_medians, initial_covariance_guess, t_dof
)

# Compute the log-likelihood of the MLE covariance matrix:
fixed_point_mle_ll = mv_t_samples_log_likelihood(
    mv_t_samples, sample_medians, fixed_point_mle_cov_matrix, t_dof
)

newton_mle_ll = mv_t_samples_log_likelihood(
    mv_t_samples, sample_medians, newton_mle_cov_matrix, t_dof
)


# Set up benchmarking for the fixed point iteration and Newton's method:
def benchmark_fixed_point():
    return mv_t_fixed_point_mle_covariance_matrix(
        centered_samples, t_dof, reverse_tol=1.0e-6, max_iter=50
    )


def benchmark_newton():
    return mv_t_newton_mle_covariance_matrix(
        centered_samples,
        t_dof,
        reverse_tol=1.0e-6,
        max_iter=50,
        # retraction=spd_manifold.retraction,
    )


# Number of repeats for timing
n_repeats = 100

# Lists to store individual iteration times
fixed_point_times = []
newton_times = []

# Time each iteration individually
for _ in range(n_repeats):
    start_time = timeit.default_timer()
    benchmark_fixed_point()
    fixed_point_times.append(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    benchmark_newton()
    newton_times.append(timeit.default_timer() - start_time)

# Calculate standard deviations
fixed_point_std = np.std(fixed_point_times)
newton_std = np.std(newton_times)

fixed_point_mean = np.mean(fixed_point_times)
newton_mean = np.mean(newton_times)

print(
    f"Fixed point mean time: {fixed_point_mean:>10.4e} +- {fixed_point_std:.4e} seconds"
)
print(f"Newton mean time:      {newton_mean:>10.4e} +- {newton_std:.4e} seconds")

# # Time the fixed point method
# fixed_point_time = timeit.timeit(benchmark_fixed_point, number=n_repeats)
# newton_time = timeit.timeit(benchmark_newton, number=n_repeats)
# print(f"Fixed point average time: {fixed_point_time/n_repeats:.6f} seconds")
# print(f"Newton method average time: {newton_time/n_repeats:.6f} seconds")

print(f"Fixed point iterations: {num_fixed_point_iters}")
print(f"Newton iterations: {num_newton_iters}")

print(f"True params log-likelihood: {true_param_ll:.2f}")
print(f"True covariance log-likelihood: {true_cov_ll:.2f}")
print(f"Initial covariance log-likelihood: {initial_coviariance_mle_ll:.2f}")
print(f"Fixed point log-likelihood: {fixed_point_mle_ll:.2f}")
print(f"Newton method log-likelihood: {newton_mle_ll:.2f}")

# %%
