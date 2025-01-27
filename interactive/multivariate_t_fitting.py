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
    squared_norms = np.sum(centered_samples * centered_samples, axis=1)
    z_bar = np.log(squared_norms[squared_norms > 1.0e-12]).mean()
    log_alpha = z_bar - np.log(dof) + digamma(0.5 * dof) - digamma(p / 2.0)
    return np.exp(log_alpha)


@jax.jit
def mle_cov_residual_chol_solve_jax(
    cov_2d: np.ndarray,
    dof: float,
    centered_samples: np.ndarray,
    num_zeroed_samples: int = 0,
) -> np.ndarray:
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape
    num_effective_samples = n - num_zeroed_samples

    inv_cov_2d = jsp.linalg.solve(cov_2d, jnp.eye(p), assume_a="pos")
    z_scores = jnp.einsum("ij,jk,ik->i", centered_samples, inv_cov_2d, centered_samples)

    scales = (p + dof) / (dof + z_scores)
    weighted_samples = centered_samples * scales[:, jnp.newaxis]
    reconstructed_cov = (weighted_samples.T @ centered_samples) / num_effective_samples

    return cov_2d - reconstructed_cov


@jax.jit
def mle_1d_cov_residual_chol_solve_jax(
    cov_1d: np.ndarray,
    dof: float,
    centered_samples: np.ndarray,
    num_zeroed_samples: int = 0,
) -> np.ndarray:
    """Compute the MLE covariance residual for a mv_t distribution."""
    num_samples, p = centered_samples.shape
    effective_num_samples = num_samples - num_zeroed_samples
    cov_2d = cov_1d.reshape(p, p)

    inv_cov_2d = jsp.linalg.solve(cov_2d, jnp.eye(p), assume_a="pos")

    # Compute 'n' quadratic forms: (n, p) @ (p, p) @ (p, n) -> (n,)
    z_scores = jnp.einsum(
        "ij,jk,ki->i", centered_samples, inv_cov_2d, centered_samples.T
    )

    scales = (p + dof) / (dof + z_scores)
    weighted_samples = centered_samples * scales[:, jnp.newaxis]
    reconstructed_cov = (weighted_samples.T @ centered_samples) / effective_num_samples

    cov_diff = cov_2d - reconstructed_cov
    return cov_diff.reshape(-1)


@jax.jit
def mle_1d_cov_residual_jax(
    cov_1d: np.ndarray, dof: float, centered_samples: np.ndarray
):
    """Compute the MLE covariance residual for a mv_t distribution.

    Original version of the MLE covariance residual function.
    """
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


def mv_t_newton_mle_covariance_matrix_old(
    centered_samples: np.ndarray,
    t_dof: float,
    retraction: Callable = spd_Bures_Wasserstein_exponential,
    reverse_tol: float = 1.0e-3,
    max_iter: int = 20,
    num_zeroed_samples: int = 0,
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
    num_effective_samples = n - num_zeroed_samples
    covariance_2d: np.ndarray
    covariance_2d = (centered_samples.T @ centered_samples) / num_effective_samples

    alpha_estimate = estimate_mle_cov_scale(centered_samples, t_dof)
    contraction_estimate = alpha_estimate * p / np.trace(sample_covariance_matrix)
    covariance_2d *= contraction_estimate

    for i in range(max_iter):
        # Compute residual and its norm
        flat_residual = mle_1d_cov_residual(
            covariance_2d.reshape(-1),
            t_dof,
            centered_samples,
            num_zeroed_samples=num_zeroed_samples,
        )
        residual_norm = la.norm(flat_residual)
        # print(f"Iteration {i}, residual norm: {residual_norm}")
        if residual_norm < reverse_tol:
            break

        flat_cov_jacobian = jac_mle_1d_cov_residual(
            covariance_2d.reshape(-1),
            t_dof,
            centered_samples,
            num_zeroed_samples=num_zeroed_samples,
        )

        # Solve for Newton step:
        newton_step: np.ndarray
        newton_step = la.solve(flat_cov_jacobian, flat_residual)

        # Retract the step back to the manifold:
        tangent_matrix = newton_step.reshape(p, p)
        tangent_matrix = 0.5 * (tangent_matrix + tangent_matrix.T)

        new_cov = retraction(covariance_2d, -tangent_matrix)
        covariance_2d = new_cov

    return covariance_2d, i


def mv_t_newton_mle_covariance_matrix(
    centered_samples: np.ndarray,
    t_dof: float,
    retraction: Callable = spd_Bures_Wasserstein_exponential,
    reverse_tol: float = 1.0e-3,
    max_iter: int = 20,
    num_zeroed_samples: int = 0,
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
    num_effective_samples = n - num_zeroed_samples

    covariance_2d: np.ndarray
    covariance_2d = (centered_samples.T @ centered_samples) / num_effective_samples

    alpha_estimate = estimate_mle_cov_scale(centered_samples, t_dof)
    contraction_estimate = alpha_estimate * (p / np.trace(covariance_2d))
    covariance_2d *= contraction_estimate

    mle_covariance, iteration_count = mv_t_newton_iterations(
        covariance_2d,
        centered_samples,
        t_dof,
        max_iter=max_iter,
        retraction=retraction,
        reverse_tol=reverse_tol,
    )

    return mle_covariance, iteration_count


def mv_t_newton_iterations(
    initial_cov_matrix: np.ndarray,
    centered_samples: np.ndarray,
    t_dof,
    num_zeroed_samples=0,
    max_iter=20,
    retraction=spd_Bures_Wasserstein_exponential,
    reverse_tol=1.0e-3,
):
    """Perform a single iteration of the Newton method for the MLE covariance matrix."""
    assert initial_cov_matrix.ndim == 2
    p, p_2 = initial_cov_matrix.shape
    assert p == p_2

    # Ensure the initial covariance matrix is positive definite, assuming the initial
    # covariance matrix estimate is guaranteed to be positive semi-definite.
    initial_cov_matrix += 1.0e-3 * np.eye(p)

    covariance_2d = initial_cov_matrix
    for i in range(max_iter):
        # Compute residual and its norm
        flat_residual = mle_1d_cov_residual(
            covariance_2d.reshape(-1),
            t_dof,
            centered_samples,
            num_zeroed_samples=num_zeroed_samples,
        )
        residual_norm = la.norm(flat_residual)
        if residual_norm < reverse_tol:
            break

        flat_cov_jacobian = jac_mle_1d_cov_residual(
            covariance_2d.reshape(-1),
            t_dof,
            centered_samples,
            num_zeroed_samples=num_zeroed_samples,
        )

        # Solve for Newton  step:
        newton_step: np.ndarray
        newton_step = sla.solve(
            flat_cov_jacobian,
            flat_residual,
            overwrite_a=True,
            overwrite_b=True,
        )

        # Retract the step back to the manifold:
        tangent_matrix = newton_step.reshape(p, p)
        tangent_matrix = 0.5 * (tangent_matrix + tangent_matrix.T)

        new_cov = retraction(covariance_2d, -tangent_matrix)
        covariance_2d = new_cov

    return covariance_2d, i


# %%
@jax.jit
def mle_cov_fixed_point_jax(
    cov_2d: np.ndarray,
    dof: float,
    centered_samples: np.ndarray,
    num_zeroed_samples: int = 0,
):
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape

    # Subtract the number of 'zeroed' samples:
    effective_num_samples = n - num_zeroed_samples

    inv_cov_2d = jsp.linalg.solve(cov_2d, jnp.eye(p), assume_a="pos")
    z_scores = jnp.einsum("ij,jk,ik->i", centered_samples, inv_cov_2d, centered_samples)

    scales = (p + dof) / (dof + z_scores)
    weighted_samples = centered_samples * scales[:, jnp.newaxis]
    reconstructed_cov = (weighted_samples.T @ centered_samples) / effective_num_samples

    return reconstructed_cov


# Function to compute the MLE covariance matrix for a multivariate t-distribution,
# using fixed point iteration:
def mv_t_fixed_point_mle_covariance_matrix(
    centered_samples: np.ndarray,
    t_dof: float,
    reverse_tol: float = 1.0e-3,
    max_iter: int = 50,
    num_zeroed_samples: int = 0,
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
    num_samples = n - num_zeroed_samples

    # Initialize the covariance matrix:
    covariance_matrix = (centered_samples.T @ centered_samples) / num_samples
    alpha_estimate = estimate_mle_cov_scale(centered_samples, t_dof)
    contraction_estimate = alpha_estimate * p / np.trace(covariance_matrix)
    covariance_matrix *= contraction_estimate

    covariance_matrix, iteration = mv_t_fixed_point_iterations(
        covariance_matrix,
        centered_samples,
        t_dof,
        num_zeroed_samples=num_zeroed_samples,
        max_iter=max_iter,
        reverse_tol=reverse_tol,
    )

    return covariance_matrix, iteration


def mv_t_fixed_point_iterations(
    initial_cov_matrix: np.ndarray,
    centered_samples: np.ndarray,
    t_dof: float,
    num_zeroed_samples: int = 0,
    max_iter: int = 20,
    reverse_tol: float = 1.0e-3,
) -> np.ndarray:
    """Perform iterations of the fixed point method for the MLE covariance matrix."""
    covariance_matrix = initial_cov_matrix.copy()
    temp_cov_matrix = initial_cov_matrix.copy()

    # Compute the MLE covariance matrix using fixed point iteration:
    for iteration in range(max_iter):
        temp_cov_matrix = mle_cov_fixed_point_jax(
            covariance_matrix,
            t_dof,
            centered_samples,
            num_zeroed_samples=num_zeroed_samples,
        )
        residual = la.norm(temp_cov_matrix - covariance_matrix, ord="fro")

        covariance_matrix = temp_cov_matrix.copy()
        if residual < reverse_tol:
            break

    return covariance_matrix, iteration


# %% Estimate the degrees of freedom of the multivariate t-distribution:
def estimate_mv_t_dof(centered_samples: np.ndarray) -> float:
    """Estimate the degrees of freedom of a multivariate t-distribution.

    From: A Novel Parameter Estimation Algorithm for the Multivariate
          t-Distribution and Its Application to Computer Vision.
    """
    p = centered_samples.shape[1]

    log_norm_sq = np.log(np.sum(centered_samples * centered_samples, axis=1))
    log_norm_sq_var = log_norm_sq.var(ddof=0)

    b = log_norm_sq_var - polygamma(1, p / 2.0)
    t_dof = (1 + np.sqrt(1 + 4 * b)) / b

    return t_dof


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


# %% Data driven estimation of the degrees of freedom of a multivariate t-distribution:
def automatic_dda_estimate_mv_t_dof(
    centered_samples: np.ndarray, rel_tol=5.0e-2, abs_tol=1.0e-1, max_iter=5
) -> float:
    """Algorithm 1: Automatic data-adaptive computation of the d.o.f. parameter nu.

    From:
    Shrinking the eigenvalues of M-estimators of covariance matrix.
    """
    # Initialize the degrees of freedom estimate:
    t_dof = kurtosis_t_dof_estimate(centered_samples)

    n = centered_samples.shape[0]
    sample_covariance_estimate = centered_samples.T @ centered_samples / n

    mle_cov_estimate, num_iters = mv_t_newton_mle_covariance_matrix(
        centered_samples, t_dof
    )
    for i in range(max_iter):
        nu_i = np.trace(sample_covariance_estimate) / np.trace(mle_cov_estimate)
        old_t_dof = t_dof
        t_dof = 2 * nu_i / max((nu_i - 1), 1.0e-5)
        print("Iteration:", i, f"Old Degrees of freedom: {old_t_dof:.2f}")
        print("Iteration:", i, f"New Degrees of freedom: {t_dof:.2f}")
        mle_cov_estimate, num_iters = mv_t_newton_iterations(
            mle_cov_estimate, centered_samples, t_dof, max_iter=5
        )

        absolute_dof_diff = np.abs(t_dof - old_t_dof)
        rel_tol_criterion = absolute_dof_diff / old_t_dof < rel_tol
        abs_tol_criterion = absolute_dof_diff < abs_tol
        if rel_tol_criterion or abs_tol_criterion:
            break

    return t_dof


# Compute a MLE mv-t covariance matrix when holding out
# each of the samples in turn:
def pop_t_dof_estimate_iteration(
    centered_samples: np.ndarray,
    t_dof: float,
    dof_rel_tol=1.0e-2,
    use_newton_mle_cov_estimates=True,
):
    """Compute the MLE covariance matrix of mv t-distribution, leaving out each sample.

    From:
    Improved estimation of the degree of freedom parameter of mv t-distribution.
    """
    num_samples, sample_dimension = centered_samples.shape

    sample_covariance_matrix = (centered_samples.T @ centered_samples) / num_samples
    grand_mle_cov_matrix, num_grand_iters = mv_t_newton_mle_covariance_matrix(
        centered_samples=centered_samples, t_dof=t_dof, max_iter=10
    )
    contraction_estimate = np.trace(grand_mle_cov_matrix) / np.trace(
        sample_covariance_matrix
    )

    loo_cov_matrices = np.zeros((num_samples, sample_dimension, sample_dimension))
    loo_sample = np.zeros((sample_dimension, 1))

    mle_cov_estimates_rel_tol = dof_rel_tol**2
    all_inner_iters = np.zeros(num_samples)
    total_loo_score = 0.0
    for i in range(num_samples):
        # Extract the leave-one-out sample as a column vector:
        loo_sample[:] = centered_samples[i, :].reshape(-1, 1)

        # Initial estimate of the leave-one-out covariance matrix,
        # subtracting the contribution of the leave-one-out sample:
        loo_cov_estimate = grand_mle_cov_matrix - contraction_estimate * (
            (centered_samples[i, :, None] @ centered_samples[i, None, :]) / num_samples
        )

        # Zero out the leave-one-out sample:
        centered_samples[i, :] = 0.0

        if use_newton_mle_cov_estimates:
            loo_cov_matrices[i], inner_iters = mv_t_newton_iterations(
                initial_cov_matrix=loo_cov_estimate,
                centered_samples=centered_samples,
                t_dof=t_dof,
                max_iter=10,
                num_zeroed_samples=1,
                reverse_tol=mle_cov_estimates_rel_tol,
            )
        else:
            loo_cov_matrices[i], inner_iters = mv_t_fixed_point_iterations(
                initial_cov_matrix=loo_cov_estimate,
                centered_samples=centered_samples,
                t_dof=t_dof,
                num_zeroed_samples=1,
                reverse_tol=mle_cov_estimates_rel_tol,
            )

        all_inner_iters[i] = inner_iters

        loo_score = (
            loo_sample.T @ sla.solve(loo_cov_matrices[i], loo_sample, assume_a="pos")
        )[0, 0]
        total_loo_score += loo_score

        # loo_fp_cov_matrix = mle_cov_fixed_point_jax(
        #     loo_cov_estimate, t_dof, centered_samples, num_zeroed_samples=1
        # )
        # loo_newton_cov_matrix, _ = mv_t_newton_iterations(
        #     loo_cov_estimate, centered_samples, t_dof, max_iter=1,
        #     num_zeroed_samples=1
        # )
        # loo_newton_fp_matrix = mle_cov_fixed_point_jax(
        #     loo_newton_cov_matrix, t_dof, centered_samples, num_zeroed_samples=1
        # )
        # exact_loo_cov_matrix_newton, num_exact_newton_iterations = (
        #     mv_t_newton_iterations(
        #         loo_newton_cov_matrix,
        #         centered_samples,
        #         t_dof,
        #         num_zeroed_samples=1,
        #         max_iter=10,
        #         reverse_tol=1.0e-6,
        #     )
        # )

        # exact_loo_cov_matrix_fp, num_exact_fp_iterations = (
        #     mv_t_fixed_point_mle_covariance_matrix(
        #         centered_samples, dof=t_dof, num_zeroed_samples=1, reverse_tol=1.0e-6
        #     )
        # )

        # fp_residual = la.norm(
        #     loo_fp_cov_matrix - exact_loo_cov_matrix_newton, ord="fro"
        # )
        # newton_residual = la.norm(
        #     loo_newton_cov_matrix - exact_loo_cov_matrix_newton, ord="fro"
        # )
        # fixed_point_residuals[i] = fp_residual
        # newton_residuals[i] = newton_residual

        # print(f"Number of exact Newton iterations: {num_exact_newton_iterations}")
        # print(f"Fixed point residual for sample {i}: {fp_residual:.2e}")
        # print(f"Newton residual for sample {i}:      {newton_residual:.2e}")

        # loo_cov_matrices[i] = loo_fp_cov_matrix

        # Restore the leave-one-out sample:
        centered_samples[i, :] = loo_sample[:].reshape(-1)

    theta = (1 - sample_dimension / num_samples) * (
        (total_loo_score / sample_dimension) / num_samples
    )
    new_t_dof = 2 * theta / (theta - 1)
    print(f"New degrees of freedom estimate: {new_t_dof:.2f}")

    avg_inner_iters = all_inner_iters.mean()
    std_inner_iters = all_inner_iters.std()
    print(f"Average inner iterations: {avg_inner_iters:.2f} +- {std_inner_iters:.2f}")

    return new_t_dof


# %% Generate some data from a multivariate t-distribution:
# np.random.seed(42)
np.random.seed(40)

n_samples = 1000
p = 5
t_dof = 5.0

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

# %% Estimate the degrees of freedom of the multivariate t-distribution:
estimated_t_dof = estimate_mv_t_dof(centered_samples)
kurtosis_t_dof = kurtosis_t_dof_estimate(centered_samples)
avg_t_dof = (estimated_t_dof + kurtosis_t_dof) / 2.0
geo_mean_t_dof = np.sqrt(estimated_t_dof * kurtosis_t_dof)

automatic_dda_t_dof = automatic_dda_estimate_mv_t_dof(centered_samples)

print(f"True degrees of freedom: {t_dof}")
print(f"(visual) Estimated degrees of freedom: {estimated_t_dof}")
print(f"Kurtosis-based estimate: {kurtosis_t_dof}")
print(f"Average estimate: {avg_t_dof}")

# Cheap and good compromise between the two estimates:
# One estimate is biased up for low dof (kurtosis),
# and the other is biased down at high dof (visual).
print(f"Geometric mean estimate: {geo_mean_t_dof}")

print(f"Automatic DDA estimate: {automatic_dda_t_dof}")

# %% Check that we can recover the "loo" covariance matrix estimates:
loo_index = 12
loo_centered_samples = centered_samples.copy()
loo_centered_samples = np.delete(loo_centered_samples, loo_index, axis=0)


# %% Compute LOO covariance matrix estimates:
# %%timeit
use_newton_mle_cov_estimates = True
t_dof_tolerance = 1.0e-2
initial_t_dof_estimate = geo_mean_t_dof
for i in range(10):
    new_t_dof = pop_t_dof_estimate_iteration(
        centered_samples,
        initial_t_dof_estimate,
        dof_rel_tol=t_dof_tolerance,
        use_newton_mle_cov_estimates=use_newton_mle_cov_estimates,
    )
    if np.abs(new_t_dof - initial_t_dof_estimate) < 1.0e-2:
        print(f"Converged to degrees of freedom: {new_t_dof:.4f}")
        break

    initial_t_dof_estimate = new_t_dof

# pop_t_dof_estimate_iteration(
#     centered_samples, geo_mean_t_dof, use_newton_mle_cov_estimates=True
# )


# %%
def mv_t_samples_log_likelihood(
    samples: np.ndarray, mean: np.ndarray, cov: np.ndarray, dof: float
):
    """Compute the log-likelihood of i.i.d. samples from a t-distribution."""
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
    """Benchmark the Fixed point method."""
    return mv_t_fixed_point_mle_covariance_matrix(
        centered_samples, t_dof, reverse_tol=1.0e-6, max_iter=50
    )


def benchmark_newton():
    """Benchmark the Newton method for the MLE covariance matrix."""
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
