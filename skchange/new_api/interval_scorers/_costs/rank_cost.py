"""Rank-based multivariate cost."""

__author__ = ["johannvk", "Tveten"]

import numpy as np
from scipy.linalg import pinvh
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseCost
from skchange.new_api.penalties import bic_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit


@njit
def _rank_cost(
    starts: np.ndarray,
    ends: np.ndarray,
    centered_data_ranks: np.ndarray,
    pinv_rank_cov: np.ndarray,
) -> np.ndarray:
    """Compute the rank cost for each segment.

    Parameters
    ----------
    starts, ends : np.ndarray
        Segment boundaries (inclusive start, exclusive end).
    centered_data_ranks : np.ndarray
        Centered data ranks of shape (n_samples, n_features).
    pinv_rank_cov : np.ndarray
        Pseudo-inverse of the rank covariance matrix.

    Returns
    -------
    costs : np.ndarray of shape (n_intervals, 1)
    """
    n_samples, n_variables = centered_data_ranks.shape
    n_intervals = starts.shape[0]
    costs = np.zeros((n_intervals, 1))

    mean_segment_ranks = np.zeros(n_variables)
    normalization_constant = 4.0 / np.square(n_samples)

    for i in range(n_intervals):
        for var in range(n_variables):
            mean_segment_ranks[var] = np.mean(
                centered_data_ranks[starts[i] : ends[i], var]
            )
        rank_score = (ends[i] - starts[i]) * (
            mean_segment_ranks.T @ pinv_rank_cov @ mean_segment_ranks
        )
        costs[i, 0] = -rank_score * normalization_constant

    return costs


def _compute_centered_ranks(X: np.ndarray) -> np.ndarray:
    """Compute centered data ranks for a data array.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)

    Returns
    -------
    centered_data_ranks : np.ndarray of shape (n_samples, n_features)
    """
    n_samples, n_variables = X.shape
    X_sorted = np.sort(X, axis=0)
    data_ranks = np.zeros_like(X, dtype=np.float64)

    for col in range(n_variables):
        # Average lower and upper ranks to handle ties correctly.
        lower = np.searchsorted(X_sorted[:, col], X[:, col], side="left")
        upper = np.searchsorted(X_sorted[:, col], X[:, col], side="right")
        data_ranks[:, col] = (1 + lower + upper) / 2.0

    return data_ranks - (n_samples + 1) / 2.0


def _compute_ranks_and_pinv_cdf_cov(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute centered data ranks and pseudo-inverse of the CDF covariance.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)

    Returns
    -------
    centered_data_ranks : np.ndarray of shape (n_samples, n_features)
    pinv_cdf_cov : np.ndarray of shape (n_features, n_features)
    """
    n_samples = X.shape[0]
    centered_data_ranks = _compute_centered_ranks(X)

    cdf_values = (centered_data_ranks + (n_samples + 1) / 2.0) / n_samples
    centered_cdf = cdf_values - 0.5
    cdf_cov = 4.0 * (centered_cdf.T @ centered_cdf) / n_samples
    pinv_cdf_cov = pinvh(cdf_cov)

    return centered_data_ranks, pinv_cdf_cov


class RankCost(BaseCost):
    r"""Rank-based multivariate cost.

    Detects changes in the distribution of multivariate data using mean rank
    statistics. The cost for a segment is the negative weighted squared norm of
    the segment's mean rank vector, where the weight matrix is the pseudo-inverse
    of the covariance of the empirical CDF [1]_:

    .. math::
        C(X_{s:e}) = -\frac{4}{n^2}(e - s)\,\bar{r}_{s:e}^T
        \hat{\Sigma}_{\text{CDF}}^+ \bar{r}_{s:e}

    where :math:`\bar{r}_{s:e}` is the vector of mean (centered) ranks in the
    segment and :math:`\hat{\Sigma}_{\text{CDF}}^+` is the pseudo-inverse of the
    empirical-CDF covariance estimated from the training data.

    The score is inherently aggregated over all features - it returns a single
    value per interval, not one per feature.

    Notes
    -----
    Satisfies the PELT sub-additivity assumption with :math:`K = 0`.

    References
    ----------
    .. [1] Lung-Yut-Fong, A., Levy-Leduc, C., & Cappe, O. (2015). Homogeneity
       and change-point detection tests for multivariate data using rank
       statistics. Journal de la societe francaise de statistique, 156(4),
       133-162.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import RankCost
    >>> X = np.random.default_rng(0).normal(size=(100, 3))
    >>> cost = RankCost()
    >>> cost.fit(X)
    RankCost()
    >>> cache = cost.precompute(X)
    >>> cost.evaluate(cache, np.array([[0, 50], [50, 100]]))
    """

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags marking this scorer as aggregated with non-positive scores."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.aggregated = True
        tags.interval_scorer_tags.non_negative_scores = False
        return tags

    @property
    def min_size(self) -> int:
        """Minimum segment size."""
        return 2

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit the cost by precomputing rank statistics from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : RankCost
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)
        self._centered_data_ranks_, self._pinv_rank_cov_ = (
            _compute_ranks_and_pinv_cdf_cov(X)
        )
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Store the data ranks for segment-wise evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            Cached rank arrays.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        return {
            "centered_data_ranks": _compute_centered_ranks(X),
            "pinv_rank_cov": self._pinv_rank_cov_,
        }

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate the rank cost on intervals.

        Parameters
        ----------
        cache : dict
            Output from :meth:`precompute`.
        interval_specs : array-like of shape (n_intervals, 2)
            Each row is ``[start, end)`` defining a segment.

        Returns
        -------
        costs : ndarray of shape (n_intervals, 1)
            Rank cost for each interval.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(interval_specs, self.interval_specs_ncols)
        starts = interval_specs[:, 0]
        ends = interval_specs[:, 1]
        return _rank_cost(
            starts,
            ends,
            cache["centered_data_ranks"],
            cache["pinv_rank_cov"],
        )

    def get_default_penalty(self) -> float:
        """Get the default BIC penalty for the fitted cost.

        The model has :math:`p(p+1)/2` free parameters (upper triangle of the
        covariance matrix).

        Returns
        -------
        float
            BIC penalty value.
        """
        check_is_fitted(self)
        p = self.n_features_in_
        n_params = p * (p + 1) // 2
        return bic_penalty(self.n_samples_in_, n_params)
