"""Example of fitting a multivariate t-distribution to data."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpy.linalg as la
import pymanopt as pm
import scipy.linalg as sla
import scipy.stats as st
from jax import jacfwd

np.random.seed(42)

n_samples = 1000
p = 3
t_dof = 3

# Spd covariance matrix:
random_nudge = np.random.randn(p).reshape(-1, 1)
cov = np.eye(p) + 0.5 * random_nudge @ random_nudge.T
la.cholesky(cov)
la.eigvalsh(cov)

mean = np.arange(p) * (-1 * np.ones(p)).cumprod()

mv_t_dist = st.multivariate_t(loc=mean, shape=cov, df=t_dof)
mv_t_samples = mv_t_dist.rvs(n_samples)

sample_medians = np.median(mv_t_samples, axis=0)
centered_samples = mv_t_samples - sample_medians


def mv_t_samples_log_likelihood(
    samples: np.ndarray, mean: np.ndarray, cov: np.ndarray, dof: float
):
    """Compute the log-likelihood of a set of samples under a mv t-distribution."""
    mv_t_dist = st.multivariate_t(loc=mean, shape=cov, df=dof)
    return mv_t_dist.logpdf(samples).sum()


true_param_ll = mv_t_samples_log_likelihood(mv_t_samples, mean, cov, t_dof)


# %%
def mle_cov_residual_np(
    cov_2d: np.ndarray, dof: float, centered_samples: np.ndarray
) -> np.ndarray:
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape

    inverted_cov_2d = la.inv(cov_2d)

    z_scores = np.sum((centered_samples @ inverted_cov_2d) * centered_samples, axis=1)

    outer_prod_scales = (p + dof) / (dof + z_scores)

    reconstructed_cov = (
        (1.0 / n) * (centered_samples.T * outer_prod_scales) @ centered_samples
    )

    return cov_2d - reconstructed_cov


def mle_cov_residual_einsum(
    cov_2d: np.ndarray, dof: float, centered_samples: np.ndarray
) -> np.ndarray:
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape
    inv_cov = np.linalg.inv(cov_2d)

    z_scores: np.ndarray
    z_scores = np.einsum("ij,jk,ik->i", centered_samples, inv_cov, centered_samples)

    scales = (p + dof) / (dof + z_scores)
    weighted_samples = centered_samples * scales[:, np.newaxis]
    reconstructed_cov = weighted_samples.T @ centered_samples / n

    return cov_2d - reconstructed_cov


def mle_cov_residual_chol_solve(
    cov_2d: np.ndarray, dof: float, centered_samples: np.ndarray
) -> np.ndarray:
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape

    cov_inverse_times_samples = sla.solve(cov_2d, centered_samples.T, assume_a="pos")
    z_scores = np.sum(centered_samples * cov_inverse_times_samples.T, axis=1)

    scales = (p + dof) / (dof + z_scores)
    weighted_samples = centered_samples * scales[:, np.newaxis]
    reconstructed_cov = weighted_samples.T @ centered_samples / n

    return cov_2d - reconstructed_cov


def mle_cov_residual_chol_solve_v2(
    cov_2d: np.ndarray, dof: float, centered_samples: np.ndarray
) -> np.ndarray:
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape

    inv_cov_2d = sla.solve(cov_2d, np.eye(p), assume_a="pos")
    z_scores = np.einsum("ij,jk,ik->i", centered_samples, inv_cov_2d, centered_samples)

    scales = (p + dof) / (dof + z_scores)
    weighted_samples = centered_samples * scales[:, np.newaxis]
    reconstructed_cov = weighted_samples.T @ centered_samples / n

    return cov_2d - reconstructed_cov


@jax.jit
def mle_cov_residual_jax(cov_2d: np.ndarray, dof: float, centered_samples: np.ndarray):
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape
    inv_cov = jnp.linalg.inv(cov_2d)
    z_scores = jnp.sum(jnp.dot(centered_samples, inv_cov) * centered_samples, axis=1)
    scales = (p + dof) / (dof + z_scores)
    reconstructed_cov = (centered_samples.T * scales) @ centered_samples / n
    return cov_2d - reconstructed_cov


@jax.jit
def mle_cov_residual_einsum_jax(
    cov_2d: np.ndarray, dof: float, centered_samples: np.ndarray
):
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape
    inv_cov = jnp.linalg.inv(cov_2d)

    # Compute 'n' quadratic forms: (n, p) @ (p, p) @ (p, n) -> (n,)
    z_scores = jnp.einsum("ij,jk,ki->i", centered_samples, inv_cov, centered_samples.T)
    X: np.ndarray
    X = jsp.linalg.solve(cov_2d, centered_samples.T, assume_a="pos")
    z_scores = jnp.sum(centered_samples * X.T, axis=1)
    scales = (p + dof) / (dof + z_scores)

    reconstructed_cov = (centered_samples.T * scales) @ centered_samples / n
    return cov_2d - reconstructed_cov


@jax.jit
def mle_cov_residual_scipy_solve_jax(
    cov_2d: np.ndarray, dof: float, centered_samples: np.ndarray
):
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape

    cov_inverse_times_samples: jnp.ndarray
    cov_inverse_times_samples = jsp.linalg.solve(
        cov_2d, centered_samples.T, assume_a="pos"
    )
    z_scores = jnp.sum(centered_samples * cov_inverse_times_samples.T, axis=1)
    scales = (p + dof) / (dof + z_scores)

    reconstructed_cov = (centered_samples.T * scales) @ centered_samples / n
    return cov_2d - reconstructed_cov


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
def mle_cov_fixed_point_jax(
    cov_2d: np.ndarray, dof: float, centered_samples: np.ndarray
):
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape
    inv_cov = jnp.linalg.inv(cov_2d)

    # Compute 'n' quadratic forms: (n, p) @ (p, p) @ (p, n) -> (n,)
    z_scores = jnp.einsum("ij,jk,ki->i", centered_samples, inv_cov, centered_samples.T)
    scales = (p + dof) / (dof + z_scores)

    reconstructed_cov = (centered_samples.T * scales) @ centered_samples / n
    return reconstructed_cov


# %%
mle_cov_residual_np(cov, t_dof, centered_samples)
mle_cov_residual_einsum(cov, t_dof, centered_samples)

mle_cov_residual_chol_solve(cov, t_dof, centered_samples)
mle_cov_residual_chol_solve_v2(cov, t_dof, centered_samples)

mle_cov_residual_jax(cov, t_dof, centered_samples)
mle_cov_residual_einsum_jax(cov, t_dof, centered_samples)

mle_cov_residual_scipy_solve_jax(cov, t_dof, centered_samples)
mle_cov_residual_chol_solve_jax(cov, t_dof, centered_samples)

# %% Set up a benchmark to compare the performance of the three implementations:
import time

n_repeats = 1000

start_time = time.time()
for _ in range(n_repeats):
    mle_cov_residual_np(cov, t_dof, centered_samples)
avg_time = (time.time() - start_time) / n_repeats
print(f"Elapsed time for numpy implementation: {avg_time:.4e}")

start_time = time.time()
for _ in range(n_repeats):
    mle_cov_residual_einsum(cov, t_dof, centered_samples)
avg_time = (time.time() - start_time) / n_repeats
print(f"Elapsed time for einsum implementation: {avg_time:.4e}")

start_time = time.time()
for _ in range(n_repeats):
    mle_cov_residual_jax(cov, t_dof, centered_samples)
avg_time = (time.time() - start_time) / n_repeats
print(f"Elapsed time for jax implementation: {avg_time:.4e}")

start_time = time.time()
for _ in range(n_repeats):
    mle_cov_residual_einsum_jax(cov, t_dof, centered_samples)
avg_time = (time.time() - start_time) / n_repeats
print(f"Elapsed time for jax einsum implementation: {avg_time:.4e}")

# Really slow:
# start_time = time.time()
# for _ in range(n_repeats):
#     mle_cov_residual_scipy_solve_jax(cov, t_dof, centered_samples)
# avg_time = (time.time() - start_time) / n_repeats
# print(f"Elapsed time for jax scipy solve implementation: {avg_time:.4e}")

# start_time = time.time()
# for _ in range(n_repeats):
#     mle_cov_residual_chol_solve(cov, t_dof, centered_samples)
# avg_time = (time.time() - start_time) / n_repeats
# print(f"Elapsed time for chol solve implementation: {avg_time:.4e}")

start_time = time.time()
for _ in range(n_repeats):
    mle_cov_residual_chol_solve_v2(cov, t_dof, centered_samples)
avg_time = (time.time() - start_time) / n_repeats
print(f"Elapsed time for chol solve V2 implementation: {avg_time:.4e}")

start_time = time.time()
for _ in range(n_repeats):
    mle_cov_residual_chol_solve_jax(cov, t_dof, centered_samples)
avg_time = (time.time() - start_time) / n_repeats
print(f"Elapsed time for chol solve jax implementation: {avg_time:.4e}")


# %%
@jax.jit
def mle_cov_fixed_point(cov_2d: np.ndarray, dof: float, centered_samples: np.ndarray):
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape
    inv_cov = jnp.linalg.inv(cov_2d)
    z_scores = jnp.sum(jnp.dot(centered_samples, inv_cov) * centered_samples, axis=1)
    scales = (p + dof) / (dof + z_scores)
    reconstructed_cov = (centered_samples.T * scales) @ centered_samples / n
    return reconstructed_cov


# %% Perform fixed point iteration to solve the MLE covariance equation:
# Initialize variables
fp_start_time = time.time()
fp_cov = cov.copy()
tol = 1e-3
fp_max_iter = 100

for i in range(fp_max_iter):
    # Compute residual and its norm
    new_cov = mle_cov_fixed_point_jax(fp_cov, t_dof, centered_samples)
    residual_norm = la.norm(new_cov - fp_cov)
    # print(f"Iteration {i}, residual norm: {residual_norm}")
    if residual_norm < tol:
        break

    # Compute the new covariance matrix
    fp_cov = new_cov

fp_elapsed_time = time.time() - fp_start_time
print(f"Elapsed time for fixed point iteration: {fp_elapsed_time:.4e}")
print(f"Number of iterations: {i}")


# %%
@jax.jit
def mle_1d_cov_equation_jax(
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


@jax.jit
def mle_1d_cov_residual_einsum_jax(
    cov_1d: np.ndarray, dof: float, centered_samples: np.ndarray
):
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape
    cov_2d = cov_1d.reshape(p, p)
    inv_cov = jnp.linalg.inv(cov_2d)

    # Compute 'n' quadratic forms: (n, p) @ (p, p) @ (p, n) -> (n,)
    z_scores = jnp.einsum("ij,jk,ki->i", centered_samples, inv_cov, centered_samples.T)
    scales = (p + dof) / (dof + z_scores)

    reconstructed_cov = (centered_samples.T * scales) @ centered_samples / n
    cov_diff = cov_2d - reconstructed_cov
    return cov_diff.reshape(-1)


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
def cov_1d_only_arg_cholesky_solve(cov_1d: np.ndarray):
    """Compute the MLE covariance residual for a mv_t distribution."""
    return mle_1d_cov_residual_chol_solve_jax(cov_1d, t_dof, centered_samples)


jac_cholesky_solve = jacfwd(cov_1d_only_arg_cholesky_solve)
jac_cholesky_solve(cov.flatten())

# %%
mle_cov_residual_np(cov, t_dof, centered_samples)

# %%
mle_cov_residual_jax(cov, t_dof, centered_samples)

cov_1d = cov.reshape(-1)

flat_cov_diff = mle_1d_cov_equation_jax(
    cov_1d, dof=t_dof, centered_samples=centered_samples
)


def cov_1d_only_arg(cov_1d: np.ndarray):
    """Compute the MLE covariance residual for a mv_t distribution."""
    return mle_1d_cov_equation_jax(cov_1d, t_dof, centered_samples)


fjac_1d_cov = jacfwd(cov_1d_only_arg)

four_times_dim_jac = fjac_1d_cov(cov_1d)

# Solve for tangent matrix:
tangent_matrix_1d = la.solve(four_times_dim_jac, flat_cov_diff)

# Solution came out almost exactly symmetric, useful:
tan_mat_2d = tangent_matrix_1d.reshape(p, p)

# %% Use pymanopt to retract the tangent matrix back to the manifold:
spd_manifold = pm.manifolds.positive_definite.SymmetricPositiveDefinite(n=p)

new_cov = spd_manifold.retraction(cov, -tan_mat_2d)

old_mle_residual = mle_cov_residual_np(cov, t_dof, centered_samples)
new_mle_residual = mle_cov_residual_np(new_cov, t_dof, centered_samples)

la.norm(old_mle_residual), la.norm(new_mle_residual)

# %% Perform root finding with Newton's method, until convergence (la.norm < 1e-3):
# Initialize variables

# cov_current = cov.copy()
cov_current = centered_samples.T @ centered_samples / n_samples
tol = 1e-3
max_iter = 100

for i in range(max_iter):
    # Compute residual and its norm
    residual = mle_cov_residual_np(cov_current, t_dof, centered_samples)
    residual_norm = la.norm(residual)
    print(f"Iteration {i}, residual norm: {residual_norm}")
    if residual_norm < tol:
        break

    # Compute flat residual
    flat_residual = mle_1d_cov_equation_jax(
        cov_current.reshape(-1), t_dof, centered_samples
    )

    # Compute Jacobian using jax
    fjac = jacfwd(
        lambda cov_1d: mle_1d_cov_equation_jax(cov_1d, t_dof, centered_samples)
    )
    jacobian = fjac(cov_current.reshape(-1))

    # Solve for Newton  step:
    newton_step = la.solve(jacobian, flat_residual)

    # Retract the step back to the manifold:
    tangent_matrix = newton_step.reshape(p, p)
    new_cov = spd_manifold.retraction(cov_current, -tangent_matrix)
    cov_current = new_cov

# %%
cov

# %%
cov_current

# %% Write a class to encapsulate the MLE covariance estimation problem:

from typing import Callable


class MultivariateStudentTCovarianceEstimator:
    """Estimate the covariance matrix of a multivariate t-distribution using MLE."""

    def __init__(
        self,
        dof: float,
        samples: np.ndarray,
        retraction: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        self.dof = dof
        self.samples = samples
        self.retraction = retraction
        self.n_samples, self.dim = samples.shape
        self.mean = np.median(samples, axis=0)
        self.centered_samples = samples - self.mean
        self.cov_current = np.cov(self.centered_samples, rowvar=False)

    def fit(self, tol: float = 1e-3, max_iter: int = 100) -> np.ndarray:
        """Fit the covariance matrix using the MLE method."""
        cov_current = self.cov_current
        for i in range(max_iter):
            residual = self.mle_cov_equation(cov_current)
            res_norm = np.linalg.norm(residual)
            print(f"Iteration {i}, residual norm: {res_norm}")
            if res_norm < tol:
                break
            cov_flat = cov_current.flatten()
            residual_flat = residual.flatten()
            jacobian_func = jacfwd(
                lambda cov_flat: self.mle_cov_equation(
                    cov_flat.reshape(self.dim, self.dim)
                ).flatten()
            )
            jacobian = jacobian_func(cov_flat)
            newton_step_flat = np.linalg.solve(jacobian, residual_flat)
            tangent_matrix = newton_step_flat.reshape(self.dim, self.dim)
            cov_current = self.retraction(cov_current, -tangent_matrix)
        self.cov_current = cov_current
        return cov_current
