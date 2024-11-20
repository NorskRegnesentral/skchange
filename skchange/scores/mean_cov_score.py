"""Test statistic for differences in the mean and/or variance for multivariate data."""

__author__ = ["johannvk"]

import numpy as np

from skchange.utils.numba import njit
from skchange.utils.numba.stats import log_det_covariance


@njit
def half_integer_digamma(twice_n: int) -> float:
    """Calculate the digamma function for half integer values, i.e. `twice_n/2`.

    The digamma function is the logarithmic derivative of the gamma function.
    This function is capable of calculating the
    digamma function for half integer values.

    Source: https://en.wikipedia.org/wiki/Digamma_function

    Parameters
    ----------
    twice_n : int
        Twice the integer value.

    Returns
    -------
    res : float
        Value of the digamma function for the half integer value.
    """
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


@njit
def likelihood_ratio_expected_value(
    sequence_length: int, cut_point: int, dimension: int
) -> float:
    """Calculate the expected value of twice the negative log likelihood ratio.

    We check that the cut point is within the sequence length, and that both `k` and `n`
    are large enough relative to the dimension `p`, to ensure that the expected
    value is finite.
    Should at least have `p+1` points on each side of a split, for `p` dimensional data.

    Parameters
    ----------
    sequence_length : int
        Length of the sequence.
    cut_point : int
        Cut point of the sequence.
    dimension : int
        Dimension of the data.

    Returns
    -------
    g_k_n : float
        Expected value of twice the negative log likelihood ratio.
    """
    n, k, p = sequence_length, cut_point, dimension

    assert 0 < k < n, "Cut point `k` must be within the sequence length `n`."
    assert p > 0, "Dimension `p` must be a positive integer."
    assert k > (p + 1), "Cut point `k` must be larger than the dimension + 1."
    assert n - k > (
        p + 1
    ), "Run length after cut point `n - k` must be larger than dimension + 1."

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


@njit
def bartlett_correction(twice_negated_log_lr, sequence_length, cut_point, dimension):
    """Calculate the Bartlett correction for the twice negated log likelihood ratio.

    Parameters
    ----------
    twice_negated_log_lr : float
        Twice the negative log likelihood ratio.
    sequence_length : int
        Length of the sequence.
    cut_point : int
        Cut point of the sequence.
    dimension : int
        Dimension of the data.

    Returns
    -------
    bartlett_corr_log_lr : float
    """
    g_k_n = likelihood_ratio_expected_value(
        sequence_length=sequence_length, cut_point=cut_point, dimension=dimension
    )
    bartlett_corr_log_lr = dimension * (dimension + 3.0) * twice_negated_log_lr / g_k_n

    return bartlett_corr_log_lr


@njit
def init_mean_cov_score(X: np.ndarray) -> np.ndarray:
    """Precompute quantities for the mean_cov_score.

    Currently does nothing as the memory overhead is too large for large datasets.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.

    Returns
    -------
    X : np.ndarray
        Initialized data matrix.
    """
    return X


@njit
def _mean_cov_log_det_term(X: np.ndarray, start: int, end: int) -> float:
    """Compute the contribution of a segment to the mean_cov score.

    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the interval to test for a change in the covariance matrix.
        (Inclusive)
    end : int
        End index of the interval to test for a change in the covariance matrix.
        (Inclusive)

    Returns
    -------
    contribution : float
        Contribution of the segment to the mean and covariance cost.
    """
    log_det_cov = log_det_covariance(X[start : end + 1, :])
    if np.isnan(log_det_cov):
        raise RuntimeError(
            f"The covariance matrix of `X[{start}:{end + 1}]` is not positive definite."
            + " Quick and dirty fix: Add a tiny amount of random noise to the data."
        )

    return (end - start + 1) * log_det_cov


@njit
def _mean_cov_score(
    X: np.ndarray,
    start: int,
    end: int,
    split: int,
) -> float:
    """Calculate the score for a change in mean and variance.

    The score is computed as the Bartlett corrected log likelihood
    ratio for a test of change in mean and/or variance at the split
    point under a multivariate Gaussian model.

    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the interval to test for a change in the precision matrix.
        (Inclusive)
    end : int
        End index of the interval to test for a change in the precision matrix.
        (Inclusive)
    split : int
        Split index of the interval to test for a change in the precision matrix.
        Include the element at the index `split` in the first segment.

    Returns
    -------
    score : float
        Score (twice negative log likelihood ratio) for the change in mean and variance.
    """
    full_span_loss = _mean_cov_log_det_term(X, start, end)
    pre_split_loss = _mean_cov_log_det_term(X, start, split)
    post_split_loss = _mean_cov_log_det_term(X, split + 1, end)
    twice_negated_log_lr = full_span_loss - pre_split_loss - post_split_loss

    score = bartlett_correction(
        twice_negated_log_lr,
        sequence_length=end + 1 - start,
        cut_point=split - start,
        dimension=X.shape[1],
    )
    return score


@njit
def mean_cov_score(
    precomputed_params: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    splits: np.ndarray,
) -> np.ndarray:
    """Calculate likelihood ratio scores for a change in mean and covariance [1]_.

    References
    ----------
    .. [1] Zamba, K. D., & Hawkins, D. M. (2009). A Multivariate Change-Point Model\
    for Change in Mean Vector and/or Covariance Structure. \
    Journal of Quality Technology, 41(3), 285-303.

    Parameters
    ----------
    precomputed_params : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Precomputed parameters from `init_covariance_score`.
    starts : np.ndarray
        Start indices of the intervals to test for a change in the covariance.
    ends : np.ndarray
        End indices of the intervals to test for a change in the covariance.
    splits : np.ndarray
        Split indices of the intervals to test for a change in the covariance.

    Returns
    -------
    scores : np.ndarray
        Scores for each start, end, split combination.
    """
    X = precomputed_params
    num_splits = len(splits)

    scores = np.zeros(num_splits, dtype=np.float64)
    for split_idx in range(num_splits):
        scores[split_idx] = _mean_cov_score(
            X, starts[split_idx], ends[split_idx], splits[split_idx]
        )

    return scores
