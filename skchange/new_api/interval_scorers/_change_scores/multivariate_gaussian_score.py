"""Multivariate Gaussian change score for a change in mean and/or covariance."""

__author__ = ["johannvk"]

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseChangeScore
from skchange.new_api.interval_scorers._costs.multivariate_gaussian_cost import (
    _multivariate_gaussian_cost_mle,
)
from skchange.new_api.penalties import bic_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import check_interval_specs
from skchange.utils.numba import njit


@njit
def _half_integer_digamma(twice_n: int) -> float:
    """Calculate the digamma function for half-integer values ``twice_n / 2``.

    Parameters
    ----------
    twice_n : int
        Twice the integer value (must be positive).

    Returns
    -------
    float
        Digamma value at ``twice_n / 2``.
    """
    assert twice_n > 0, "twice_n must be a positive integer."

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
def _likelihood_ratio_expected_value(
    sequence_length: int, cut_point: int, dimension: int
) -> float:
    """Calculate the expected value of twice the negative log-likelihood ratio.

    Both ``cut_point`` and ``sequence_length - cut_point`` must exceed
    ``dimension`` so that each side yields a full-rank covariance estimate.

    Parameters
    ----------
    sequence_length : int
        Total length of the sequence.
    cut_point : int
        Position of the cut within the sequence.
    dimension : int
        Dimension of the data.

    Returns
    -------
    float
        Expected value of twice the negative log-likelihood ratio.
    """
    n, k, p = sequence_length, cut_point, dimension

    assert 0 < k < n, "Cut point k must be within the sequence length n."
    assert p > 0, "Dimension p must be a positive integer."
    assert k > p, "Cut point k must be larger than the dimension p."
    assert n - k > p, "Run length after cut point (n - k) must be larger than p."

    g_k_n = p * (
        np.log(2)
        + (n - 1) * np.log(n - 1)
        - (n - k - 1) * np.log(n - k - 1)
        - (k - 1) * np.log(k - 1)
    )

    for j in range(1, p + 1):
        g_k_n += (
            (n - 1) * _half_integer_digamma(n - j)
            - (k - 1) * _half_integer_digamma(k - j)
            - (n - k - 1) * _half_integer_digamma(n - k - j)
        )

    return g_k_n


@njit
def _compute_bartlett_corrections(
    sequence_lengths: np.ndarray, cut_points: np.ndarray, dimension: int
) -> np.ndarray:
    """Calculate the Bartlett correction factors for each interval.

    Parameters
    ----------
    sequence_lengths : np.ndarray
        Total lengths of each interval.
    cut_points : np.ndarray
        Cut points (relative to start) of each interval.
    dimension : int
        Dimension of the data.

    Returns
    -------
    np.ndarray of shape (n_intervals, 1)
        Bartlett correction factor for each interval.
    """
    bartlett_corrections = np.zeros((sequence_lengths.shape[0], 1), dtype=np.float64)

    for i in range(sequence_lengths.shape[0]):
        g_k_n = _likelihood_ratio_expected_value(
            sequence_length=sequence_lengths[i],
            cut_point=cut_points[i],
            dimension=dimension,
        )
        bartlett_corrections[i, 0] = dimension * (dimension + 3.0) / g_k_n

    return bartlett_corrections


class MultivariateGaussianScore(BaseChangeScore):
    """Multivariate Gaussian change score for a change in mean and/or covariance.

    Scores are calculated as likelihood ratio scores for a change in mean and
    covariance under a multivariate Gaussian distribution [1]_:

    .. math::
        S(X_{s:e}, b) = C(X_{s:e}) - C(X_{s:b}) - C(X_{b:e})

    where :math:`C` is :class:`MultivariateGaussianCost`.

    To stabilise the score, the Bartlett correction is applied by default,
    adjusting for the relative sizes of the left and right segments so that
    scores approach the chi-squared distribution asymptotically.

    The score is inherently aggregated over all features - it returns a single
    value per interval.

    Parameters
    ----------
    apply_bartlett_correction : bool, default=True
        Whether to apply the Bartlett correction to the change scores.

    References
    ----------
    .. [1] Zamba, K. D., & Hawkins, D. M. (2009). A Multivariate Change-Point Model
       for Change in Mean Vector and/or Covariance Structure. Journal of Quality
       Technology, 41(3), 285-303.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import MultivariateGaussianScore
    >>> X = np.random.default_rng(0).normal(size=(100, 3))
    >>> scorer = MultivariateGaussianScore()
    >>> scorer.fit(X)
    MultivariateGaussianScore()
    >>> cache = scorer.precompute(X)
    >>> scorer.evaluate(cache, np.array([[0, 25, 50], [50, 75, 100]]))
    """

    _parameter_constraints: dict = {
        "apply_bartlett_correction": ["boolean"],
    }

    def __init__(self, apply_bartlett_correction: bool = True):
        self.apply_bartlett_correction = apply_bartlett_correction

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags marking this scorer as aggregated."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.aggregated = True
        return tags

    @property
    def min_size(self) -> int:
        """Minimum sub-interval size (n_features + 1 to keep each side full-rank)."""
        check_is_fitted(self)
        return self.n_features_in_ + 1

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate the multivariate Gaussian change score on intervals.

        Parameters
        ----------
        cache : dict
            Output from :meth:`precompute`.
        interval_specs : array-like of shape (n_intervals, 3)
            Each row is ``[start, split, end)``.

        Returns
        -------
        scores : ndarray of shape (n_intervals, 1)
            Change score for each interval.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(interval_specs, self.interval_specs_ncols)
        starts = interval_specs[:, 0]
        splits = interval_specs[:, 1]
        ends = interval_specs[:, 2]

        X = cache["X"]

        left_costs = _multivariate_gaussian_cost_mle(starts, splits, X, self.min_size)
        right_costs = _multivariate_gaussian_cost_mle(splits, ends, X, self.min_size)
        full_costs = _multivariate_gaussian_cost_mle(starts, ends, X, self.min_size)

        raw_scores = full_costs - (left_costs + right_costs)

        if self.apply_bartlett_correction:
            sequence_lengths = ends - starts
            cut_points = splits - starts
            corrections = _compute_bartlett_corrections(
                sequence_lengths, cut_points, self.n_features_in_
            )
            return corrections * raw_scores
        else:
            return raw_scores

    def get_default_penalty(self) -> float:
        r"""Get the default penalty.

        When the Bartlett correction is applied, the score is approximately
        chi-squared distributed with :math:`p(p+3)/2` degrees of freedom. The
        Bartlett correction already normalises for model complexity, so only the
        changepoint location contributes to the BIC penalty, giving
        :math:`\\log(n)`.

        Without the Bartlett correction the raw likelihood ratio scale is used and
        the full BIC penalty :math:`(p + p(p+1)/2 + 1) \\log(n)` applies.

        Returns
        -------
        float
        """
        check_is_fitted(self)
        if self.apply_bartlett_correction:
            # The Bartlett correction normalises the score to approximately chi-squared
            # with p*(p+3)/2 degrees of freedom, already accounting for model
            # complexity.
            # Only the changepoint location contributes to the penalty: log(n).
            return float(np.log(self.n_samples_in_))
        p = self.n_features_in_
        n_params = p + p * (p + 1) // 2
        return bic_penalty(self.n_samples_in_, n_params)
