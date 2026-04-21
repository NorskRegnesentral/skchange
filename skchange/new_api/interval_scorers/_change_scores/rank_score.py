"""Rank-based change score for multivariate data."""

__author__ = ["johannvk"]

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseChangeScore
from skchange.new_api.interval_scorers._costs.rank_cost import (
    _compute_ranks_and_pinv_cdf_cov,
)
from skchange.new_api.penalties import bic_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import check_interval_specs, validate_data


def _compute_sorted_ranks(
    centered_data_ranks: np.ndarray,
    segment_start: int,
    segment_end: int,
    output_array: np.ndarray,
):
    """Compute sorted ranks for a given segment.

    Computes the sorted ranks for the data within the specified segment defined
    by start and end indices. The ranks are centered by subtracting the mean rank.

    Parameters
    ----------
    centered_data_ranks : np.ndarray
        The centered data ranks.
    segment_start : int
        The start index of the segment, inclusive.
    segment_end : int
        The end index of the segment, exclusive.
    output_array : np.ndarray
        The array to store the computed ranks (modified in place).
    """
    n_variables = centered_data_ranks.shape[1]
    segment_sorted_by_column = np.sort(
        centered_data_ranks[segment_start:segment_end, :], axis=0
    )

    for col in range(n_variables):
        # Upper right ranks: (a[i-1] < v <= a[i])
        output_array[0 : (segment_end - segment_start), col] = 1 + np.searchsorted(
            segment_sorted_by_column[:, col],
            centered_data_ranks[segment_start:segment_end, col],
            side="left",
        )
        # Average lower and upper ranks to handle duplicates.
        output_array[0 : (segment_end - segment_start), col] += np.searchsorted(
            segment_sorted_by_column[:, col],
            centered_data_ranks[segment_start:segment_end, col],
            side="right",
        )
        output_array[0 : (segment_end - segment_start), col] /= 2

    output_array[0 : (segment_end - segment_start), :] -= (
        (segment_end - segment_start) + 1
    ) / 2.0


def direct_rank_score(
    change_cuts: np.ndarray,
    centered_data_ranks: np.ndarray,
    pinv_rank_cov: np.ndarray,
) -> np.ndarray:
    """Compute the rank-based change score for segments.

    For each interval ``[start, split, end]``, computes the score based on the mean
    ranks before and after the split, normalized by the pseudo-inverse of the rank
    covariance matrix.

    Parameters
    ----------
    change_cuts : np.ndarray of shape (n_intervals, 3)
        Each row is ``[start, split, end]``.
    centered_data_ranks : np.ndarray
        The centered data ranks.
    pinv_rank_cov : np.ndarray
        The pseudo-inverse of the rank covariance matrix.

    Returns
    -------
    np.ndarray of shape (n_intervals,)
        Rank-based change scores for each segment.
    """
    n_variables = centered_data_ranks.shape[1]
    rank_scores = np.zeros(change_cuts.shape[0])
    if len(rank_scores) == 0:
        return rank_scores

    mean_segment_ranks = np.zeros(n_variables)

    max_interval_length = int(np.max(np.diff(change_cuts[:, [0, 2]], axis=1)))
    segment_data_ranks = np.zeros((max_interval_length, n_variables))

    prev_segment_start = change_cuts[0, 0]
    prev_segment_end = change_cuts[0, 2]

    _compute_sorted_ranks(
        centered_data_ranks,
        segment_start=prev_segment_start,
        segment_end=prev_segment_end,
        output_array=segment_data_ranks,
    )

    for i, cut in enumerate(change_cuts):
        segment_start, segment_split, segment_end = cut

        full_segment_length = segment_end - segment_start
        pre_split_length = segment_split - segment_start
        post_split_length = segment_end - segment_split

        normalization_constant = 2.0 / np.sqrt(
            full_segment_length * pre_split_length * post_split_length
        )

        if segment_start != prev_segment_start or segment_end != prev_segment_end:
            _compute_sorted_ranks(
                centered_data_ranks,
                segment_start=segment_start,
                segment_end=segment_end,
                output_array=segment_data_ranks,
            )

        if pre_split_length < post_split_length:
            mean_segment_ranks[:] = (
                -np.sum(
                    segment_data_ranks[0 : (segment_split - segment_start), :], axis=0
                )
                * normalization_constant
            )
        else:
            mean_segment_ranks[:] = (
                np.sum(
                    segment_data_ranks[0 : (segment_split - segment_start), :], axis=0
                )
                * normalization_constant
            )

        rank_scores[i] = mean_segment_ranks.T @ pinv_rank_cov @ mean_segment_ranks

        prev_segment_start = segment_start
        prev_segment_end = segment_end

    return rank_scores


class RankScore(BaseChangeScore):
    """Rank-based change score for multivariate data.

    Uses mean rank statistics to detect changes in the distribution of
    multivariate data. Scores the difference in mean ranks for each variable
    before and after the split, normalized by the pseudo-inverse of the rank
    covariance matrix [1]_.

    Requires sorting the data within each segment, leading to increased
    computational complexity per evaluation. Suitable for offline detection
    with moderate data sizes.

    References
    ----------
    .. [1] Lung-Yut-Fong, A., Lévy-Leduc, C., & Cappé, O. (2015). Homogeneity and
       change-point detection tests for multivariate data using rank statistics.
       Journal de la société française de statistique, 156(4), 133-162.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import RankScore
    >>> X = np.random.default_rng(0).normal(size=(100, 3))
    >>> scorer = RankScore()
    >>> scorer.fit(X)
    RankScore()
    >>> cache = scorer.precompute(X)
    >>> scorer.evaluate(cache, np.array([[0, 25, 50], [50, 75, 100]]))
    """

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags marking this scorer as aggregated."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.aggregated = True
        return tags

    @property
    def min_size(self) -> int:
        """Minimum interval size (2)."""
        return 2

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit the score by precomputing rank statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : RankScore
        """
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=True)
        self._centered_data_ranks_, self._pinv_rank_cov_ = (
            _compute_ranks_and_pinv_cdf_cov(X)
        )
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Return precomputed rank data for segment-wise evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data (not re-ranked; ranks are fixed from training).

        Returns
        -------
        cache : dict
        """
        check_is_fitted(self)
        validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=False)
        return {
            "centered_data_ranks": self._centered_data_ranks_,
            "pinv_rank_cov": self._pinv_rank_cov_,
        }

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate the rank-based change score on intervals.

        Parameters
        ----------
        cache : dict
            Output from :meth:`precompute`.
        interval_specs : array-like of shape (n_intervals, 3)
            Each row is ``[start, split, end)``.

        Returns
        -------
        scores : ndarray of shape (n_intervals, 1)
            Rank change score for each interval.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(interval_specs, self.interval_specs_ncols)
        scores = direct_rank_score(
            interval_specs,
            cache["centered_data_ranks"],
            cache["pinv_rank_cov"],
        )
        return scores.reshape(-1, 1)

    def get_default_penalty(self) -> float:
        """Get the default BIC penalty.

        Returns
        -------
        float
        """
        check_is_fitted(self)
        p = self.n_features_in_
        n_params = p * (p + 1) // 2
        return bic_penalty(self.n_samples_in_, n_params)
