"""Test statistic for differences in the mean and/or variance for multivariate data."""

from typing import Tuple

import numpy as np
from numba import njit
from numpy import euler_gamma
from skchange.utils.numba.general import truncate_below
from skchange.utils.numba.stats import col_cumsum


## Should at least have 'p+1' points on each side of a split, for 'p' dimensional data.
@njit(cache=True)
def half_integer_digamma(twice_n: int) -> float:
    """Calculate the digamma function for half integer values, i.e. twice_n/2.

    The digamma function is the logarithmic derivative of the gamma function.
    This function is capable of calculating the digamma function for half integer values.
    Source: https://en.wikipedia.org/wiki/Digamma_function
    """
    assert isinstance(twice_n, int), "n must be an integer."
    assert twice_n > 0, "n must be a positive integer."

    if twice_n % 2 == 0:
        # Even integer: twice_n = 2n
        res = -np.euler_gamma
        n = twice_n // 2
        for k in range(0, n - 1):
            res += 1.0 / (k + 1.0)
    else:
        res = -2 * np.log(2) - np.euler_gamma
        # Odd integer: twice_n = 2n + 1
        n = (twice_n - 1) // 2
        for k in range(1, n + 1):
            res += 2.0 / (2.0 * k - 1.0)

    return res


@njit(cache=True)
def likelihood_ratio_expected_value(
    sequence_length: int, cut_point: int, dimension: int
) -> float:
    """Calculate the expected value of twice the negative log likelihood ratio."""
    n, k, p = sequence_length, cut_point, dimension

    # Check that the cut point is within the sequence length, and that both 'k' and 'n' are
    # large enough relative to the dimension 'p', to ensure that the expected value is finite.
    assert 0 < k < n, "Cut point 'k' must be within the sequence length 'n'."
    assert p > 0, "Dimension 'p' must be a positive integer."
    assert k > (p + 1), "Cut point 'k' must be larger than the dimension + 1."
    assert n - k > (
        p + 1
    ), "Run length after cut point 'n - k' must be larger than dimension + 1."

    g_k_n = p * (
        np.log(2)
        + (n - 1) * np.log(n - 1)
        - (n - k - 1) * np.log(n - k - 1)
        - (k - 1) * np.log(k - 1)
    )

    for j in range(1, p + 1):
        g_k_n += (
            (n - 1) * half_integer_digamma(n - j)
            - (k - 1) * half_integer_digamma(k - j)
            - (n - k - 1) * half_integer_digamma(n - k - j)
        )

    return g_k_n


@njit(cache=True)
def bartlett_correction(twice_negated_log_lr, sequence_length, cut_point, dimension):
    """Calculate the Bartlett correction for the twice negated log likelihood ratio."""

    g_k_n = likelihood_ratio_expected_value(
        sequence_length=sequence_length, cut_point=cut_point, dimension=dimension
    )

    return dimension * (dimension + 3.0) * twice_negated_log_lr / g_k_n


# @njit(cache=True)
def init_multivariate_meanvar_score(X: np.ndarray) -> np.ndarray:
    """Initialize the precision matrix change point detection."""
    # TODO: Should (could) compute "rolling" covariance matrices here?
    #       - Would take up more memory, but save on computation time.
    return X


# @njit(cache=True)
def log_pdet_cov_matrix(cov: np.ndarray, zero_eigval_tol: float = 1.0e-10) -> float:
    """Calculate the log determinant of the covariance matrix."""
    eigvals = np.linalg.eigvalsh(cov)
    non_zero_eigvals = eigvals[eigvals > zero_eigval_tol]
    return np.sum(np.log(non_zero_eigvals))


def log_top_k_eigenvals_det_cov_matrix(cov: np.ndarray, top_k_eigvals: int) -> float:
    """Calculate the log determinant of the covariance matrix."""
    eigvals = np.linalg.eigvalsh(cov)
    # Sort the eigenvalues in descending order:
    eigvals[::-1].sort()
    # Extract the top 'k' eigenvalues:
    top_k_eigs = eigvals[0:top_k_eigvals]

    return np.sum(np.log(top_k_eigs))


# @njit(cache=True)
def log_det_cov_matrix(cov: np.ndarray, diag_var_perturbation: float = 0.0) -> float:
    """Calculate the determinant of the covariance matrix."""
    if diag_var_perturbation > 0.0:
        cov += diag_var_perturbation * np.eye(cov.shape[0])

    det_sign, log_abs_det = np.linalg.slogdet(cov)
    if det_sign <= 0:
        raise ValueError("Covariance matrix is not positive definite.")

    return log_abs_det


# @njit(cache=True)
def compute_log_det(
    cov_matrix, use_pseudo_det=False, diag_perturbation=0.0, zero_eigval_tol=0.0
):
    if use_pseudo_det:
        # Compute the pseudo-determinant of the covariance matrix.
        log_det = log_pdet_cov_matrix(cov_matrix, zero_eigval_tol=zero_eigval_tol)
    else:
        # complete_min_diag_var = np.min(np.diag(cov_matrix))
        log_det = log_det_cov_matrix(
            cov_matrix,
            diag_var_perturbation=diag_perturbation,
        )
    return log_det


def full_window_pre_and_post_split_log_pdet(
    full_cov, pre_split_cov, post_split_cov, zero_eigval_tol
):
    full_cov_eigvals = np.linalg.eigvalsh(full_cov)
    pre_split_eigvals = np.linalg.eigvalsh(pre_split_cov)
    post_split_eigvals = np.linalg.eigvalsh(post_split_cov)

    # Sort in descending order:
    full_cov_eigvals[::-1].sort()
    pre_split_eigvals[::-1].sort()
    post_split_eigvals[::-1].sort()

    # Find the number of non-zero eigenvalues:
    full_cov_num_nonzero_eigvals = np.sum(full_cov_eigvals > zero_eigval_tol)
    pre_split_num_nonzero_eigvals = np.sum(pre_split_eigvals > zero_eigval_tol)
    post_split_num_nonzero_eigvals = np.sum(post_split_eigvals > zero_eigval_tol)

    min_num_nonzero_eigvals = min(
        full_cov_num_nonzero_eigvals,
        pre_split_num_nonzero_eigvals,
        post_split_num_nonzero_eigvals,
    )

    # Extract the top 'k' eigenvalues:
    top_k_full_cov_eigs = full_cov_eigvals[0:min_num_nonzero_eigvals]
    top_k_pre_split_eigs = pre_split_eigvals[0:min_num_nonzero_eigvals]
    top_k_post_split_eigs = post_split_eigvals[0:min_num_nonzero_eigvals]

    # Compute the log determinant of the covariance matrices:
    log_det_full_cov = np.sum(np.log(top_k_full_cov_eigs))
    log_det_pre_split = np.sum(np.log(top_k_pre_split_eigs))
    log_det_post_split = np.sum(np.log(top_k_post_split_eigs))

    return log_det_full_cov, log_det_pre_split, log_det_post_split


# @njit(cache=True)
def _multivariate_pdet_meanvar_score(
    X: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    split: np.ndarray,
    zero_eigval_tol: float = 1e-8,
    apply_bartlett_correction: bool = False,
) -> np.ndarray:
    """Calculate the score (twice negative log likelihood ratio) for a change in mean and variance.

    Under a multivariate Gaussian model.
    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the interval to test for a change in the precision matrix.
    end : int
        End index of the interval to test for a change in the precision matrix.
    split : int
        Split index of the interval to test for a change in the precision matrix.
    """
    # TODO: Test comparing the pseudo-determinant computed with the same number of top k eigenvalues,
    # where k = min(num_nonzero_eigvals(complete_cov), num_nonzero_eigvals(pre_split_cov),
    # num_nonzero_eigvals(post_split_cov)). This will perhaps lead to more fair comparisons?

    complete_cov = np.cov(X[start : (end + 1)], rowvar=False, ddof=0)
    pre_split_cov = np.cov(X[start:split], rowvar=False, ddof=0)
    post_split_cov = np.cov(X[split : (end + 1)], rowvar=False, ddof=0)

    log_det_complete = compute_log_det(
        complete_cov, use_pseudo_det=True, zero_eigval_tol=zero_eigval_tol
    )
    log_det_pre_split = compute_log_det(
        pre_split_cov, use_pseudo_det=True, zero_eigval_tol=zero_eigval_tol
    )
    log_det_post_split = compute_log_det(
        post_split_cov, use_pseudo_det=True, zero_eigval_tol=zero_eigval_tol
    )

    # Compute the log pseudo-determinant of the covariance matrices,
    # using the same number of top 'k' eigenvalues:
    # log_det_complete, log_det_pre_split, log_det_post_split = (
    #     full_window_pre_and_post_split_log_pdet(
    #         complete_cov, pre_split_cov, post_split_cov, zero_eigval_tol
    #     )
    # )

    full_span_loss = (end - start + 1) * log_det_complete
    pre_split_loss = (split - start) * log_det_pre_split
    post_split_loss = (end + 1 - split) * log_det_post_split

    twice_negated_log_lr = full_span_loss - pre_split_loss - post_split_loss

    # The negated log likelihood ratio is theoretically always non-negative.
    # Sometimes numerical issues can cause the log likelihood ratio to be negative.
    # In these cases, we truncate the value to zero.
    if twice_negated_log_lr < 0.0:
        twice_negated_log_lr = 0.0

    if apply_bartlett_correction:
        score = bartlett_correction(
            twice_negated_log_lr,
            sequence_length=end + 1 - start,
            cut_point=split - start,
            dimension=X.shape[1],
        )
    else:
        score = twice_negated_log_lr

    return score


# @njit(cache=True)
def _multivariate_meanvar_score(
    X: np.ndarray,
    start: int,
    end: int,
    split: int,
    diag_var_perturbation: float = 1e-10,
    apply_bartlett_correction: bool = False,
) -> float:
    """Calculate the score (twice negative log likelihood ratio) for a change in mean and variance.

    Under a multivariate Gaussian model.
    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the interval to test for a change in the precision matrix. (Inclusive)
    end : int
        End index of the interval to test for a change in the precision matrix. (Inclusive)
    split : int
        Split index of the interval to test for a change in the precision matrix.
        Include the element at the index 'split' in the second segment. 
    diag_var_perturbation : float
        Diagonal variance perturbation to add to the covariance matrix.
    apply_bartlett_correction : bool
        Whether to apply the Bartlett correction to the score.
    Returns
    -------
    score : float
        Score (twice negative log likelihood ratio) for the change in mean and variance.
    """
    complete_cov = np.cov(X[start : (end + 1)], rowvar=False, ddof=0)
    log_det_complete = log_det_cov_matrix(
        complete_cov, diag_var_perturbation=diag_var_perturbation
    )
    full_span_loss = (end - start + 1) * log_det_complete

    pre_split_cov = np.cov(X[start:split], rowvar=False, ddof=0)
    log_det_pre_split = log_det_cov_matrix(
        pre_split_cov, diag_var_perturbation=diag_var_perturbation
    )
    pre_split_loss = (split - start) * log_det_pre_split

    post_split_cov = np.cov(X[split : (end + 1)], rowvar=False, ddof=0)
    log_det_post_split = log_det_cov_matrix(
        post_split_cov, diag_var_perturbation=diag_var_perturbation
    )
    post_split_loss = (end + 1 - split) * log_det_post_split

    twice_negated_log_lr = full_span_loss - pre_split_loss - post_split_loss

    if apply_bartlett_correction:
        score = bartlett_correction(
            twice_negated_log_lr,
            sequence_length=end + 1 - start,
            cut_point=split - start,
            dimension=X.shape[1],
        )
    else:
        score = twice_negated_log_lr

    return score


# @njit(cache=True)
def multivariate_pdet_meanvar_score(
    precomputed_params: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    splits: np.ndarray,
) -> np.ndarray:
    """Calculate the CUSUM score for a change in the mean and covariance for.

    Parameters
    ----------
    precomputed_params : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Precomputed parameters from init_covariance_score.
    starts : np.ndarray
        Start indices of the intervals to test for a change in the covariance.
    ends : np.ndarray
        End indices of the intervals to test for a change in the covariance.
    splits : np.ndarray
        Split indices of the intervals to test for a change in the covariance.

    Returns
    -------
    scores : np.ndarray
        Scores (-2 negative log likelihood) for each split segment.
    """
    # print("Not using pseudo-determinant for now.")
    # Data matrix, column variables:
    X = precomputed_params
    num_splits = len(splits)

    # Assume: 'start', 'end', and 'split' are 1D integer arrays,
    # of the same length.
    if not (len(starts) == len(ends) == num_splits):
        raise ValueError("Lengths of 'starts', 'ends', and 'splits' must be the same.")

    scores = np.zeros(num_splits, dtype=np.float64)
    for split_idx in range(num_splits):
        scores[split_idx] = _multivariate_pdet_meanvar_score(
            X, starts[split_idx], ends[split_idx], splits[split_idx]
        )

    return scores


# @njit(cache=True)
def multivariate_meanvar_score(
    precomputed_params: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    splits: np.ndarray,
) -> np.ndarray:
    """Calculate the CUSUM score for a change in the mean and covariance under a multivariate Gaussian model.

    Parameters
    ----------
    precomputed_params : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Precomputed parameters from init_covariance_score.
    starts : np.ndarray
        Start indices of the intervals to test for a change in the covariance.
    ends : np.ndarray
        End indices of the intervals to test for a change in the covariance.
    splits : np.ndarray
        Split indices of the intervals to test for a change in the covariance.

    References:
    - A Multivariate Change point Model for Change in Mean Vector and/or Covariance Structure: Detection of Isolated
    Systolic Hypertension (ISH). K.D. Zamba.

    Returns
    -------
    scores : np.ndarray
        Scores (-2 negative log likelihood) for each split segment.
    """
    # print("Not using pseudo-determinant for now.")
    # Data matrix, column variables:
    X = precomputed_params
    num_splits = len(splits)

    # Assume: 'start', 'end', and 'split' are 1D integer arrays,
    # of the same length.
    if not (len(starts) == len(ends) == num_splits):
        raise ValueError("Lengths of 'starts', 'ends', and 'splits' must be the same.")

    scores = np.zeros(num_splits, dtype=np.float64)
    for split_idx in range(num_splits):
        scores[split_idx] = _multivariate_meanvar_score(
            X, starts[split_idx], ends[split_idx], splits[split_idx]
        )

    return scores
