"""Rank-based change score for multivariate data.

Detects distributional changes using rank statistics, assuming continuous CDFs.
The score is based on the difference in mean ranks before and after a split,
normalized by the pseudo-inverse of the rank covariance matrix.

References
----------
- Hawkins, D. M., & Zamba, K. D. (2005). Statistical process control for shifts in mean or variance using rank statistics. Technometrics, 47(2), 164-173.
"""

import numpy as np

from ..base import BaseIntervalScorer
from ..costs._rank_cost import _compute_ranks_and_pinv_cdf_cov
from ..utils.numba import njit


# @njit
def direct_rank_score(
    change_cuts: np.ndarray,
    centered_data_ranks: np.ndarray,
    pinv_rank_cov: np.ndarray,
):
    """Compute the rank-based change score for segments.

    For each interval [start, split, end], computes the score based on the mean ranks
    before and after the split, normalized by the pseudo-inverse of the rank covariance.

    Parameters
    ----------
    segment_starts : np.ndarray
        Start indices of the segments.
    segment_splits : np.ndarray
        Split indices of the segments.
    segment_ends : np.ndarray
        End indices of the segments.
    centered_data_ranks : np.ndarray
        The centered data ranks.
    pinv_rank_cov : np.ndarray
        The pseudo-inverse of the rank covariance matrix.

    Returns
    -------
    np.ndarray
        The rank-based change scores for each segment.
    """
    n_variables = centered_data_ranks.shape[1]
    rank_scores = np.zeros(change_cuts.shape[0])
    if len(rank_scores) == 0:
        # Return early if not at least one cut to evaluate:
        return rank_scores

    mean_segment_ranks = np.zeros(n_variables)

    # Use CDF covariance pseudo-inverse computed on all the data.
    max_interval_length = np.max(np.diff(change_cuts[:, [0, 2]], axis=1))
    segment_data_ranks = np.zeros((max_interval_length, n_variables))

    prev_segment_start = change_cuts[0, 0]
    prev_segment_end = change_cuts[0, 2]

    # Rank values within the first segment:
    segment_sorted_by_column = np.sort(
        centered_data_ranks[prev_segment_start:prev_segment_end, :], axis=0
    )

    for col in range(n_variables):
        # Compute upper right ranks: (a[i-1] < v <= a[i])
        # One-indexed lower ranks:
        segment_data_ranks[0 : (prev_segment_end - prev_segment_start), col] = (
            1
            + np.searchsorted(
                segment_sorted_by_column[:, col],
                centered_data_ranks[prev_segment_start:prev_segment_end, col],
                side="left",
            )
        )

        # Must average lower and upper ranks to work on data with duplicates:
        # Add lower left ranks: (a[i-1] <= v < a[i])
        segment_data_ranks[0 : (prev_segment_end - prev_segment_start), col] += (
            np.searchsorted(
                segment_sorted_by_column[:, col],
                centered_data_ranks[prev_segment_start:prev_segment_end, col],
                side="right",
            )
        )
        segment_data_ranks[0 : (prev_segment_end - prev_segment_start), col] /= 2

    segment_data_ranks[0 : (prev_segment_end - prev_segment_start), :] -= (
        (prev_segment_end - prev_segment_start) + 1
    ) / 2.0

    for i, cut in enumerate(change_cuts):
        # Unpack cut row into start, split, end:
        segment_start, segment_split, segment_end = cut

        full_segment_length = segment_end - segment_start
        pre_split_length = segment_split - segment_start
        post_split_length = segment_end - segment_split

        normalization_constant = 2.0 / np.sqrt(
            full_segment_length * pre_split_length * post_split_length
        )

        if segment_start != prev_segment_start or segment_end != prev_segment_end:
            # Rank values within the segment:
            segment_sorted_by_column = np.sort(
                centered_data_ranks[segment_start:segment_end, :], axis=0
            )

            for col in range(n_variables):
                # Compute upper right ranks: (a[i-1] < v <= a[i])
                # One-indexed lower ranks:
                segment_data_ranks[0 : (segment_end - segment_start), col] = (
                    1
                    + np.searchsorted(
                        segment_sorted_by_column[:, col],
                        centered_data_ranks[segment_start:segment_end, col],
                        side="left",
                    )
                )

                # Must average lower and upper ranks to work on data with duplicates:
                # Add lower left ranks: (a[i-1] <= v < a[i])
                segment_data_ranks[0 : (segment_end - segment_start), col] += (
                    np.searchsorted(
                        segment_sorted_by_column[:, col],
                        centered_data_ranks[segment_start:segment_end, col],
                        side="right",
                    )
                )
                segment_data_ranks[0 : (segment_end - segment_start), col] /= 2

            segment_data_ranks[0 : (segment_end - segment_start), :] -= (
                full_segment_length + 1
            ) / 2.0

        # Use score formulation from first part centered ranks:
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

    return rank_scores


class RankScore(BaseIntervalScorer):
    """Rank-based change score for multivariate data.

    Detects distributional changes using rank statistics, assuming continuous CDFs.
    The score is based on the difference in mean ranks before and after a split,
    normalized by the pseudo-inverse of the rank covariance matrix.

    Parameters
    ----------
    None

    Notes
    -----
    - The score is inherently multivariate and aggregated.
    - Suitable for detecting changes in mean or variance in non-Gaussian data.

    References
    ----------
    .. [1] Lung-Yut-Fong, A., Lévy-Leduc, C., & Cappé, O. (2015). Homogeneity and
       change-point detection tests for multivariate data using rank statistics.
       Journal de la société française de statistique, 156(4), 133-162.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "task": "change_score",
        "distribution_type": "None",
        "is_conditional": False,
        "is_aggregated": True,
        "is_penalised": False,
    }

    def __init__(self):
        super().__init__()

    def _fit(self, X: np.ndarray, y=None):
        """Fit the score.

        Precomputes rank statistics and covariance matrix for efficient evaluation.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y: None
            Ignored. Included for API consistency by convention.
        """
        self._centered_data_ranks, self._pinv_rank_cov = (
            _compute_ranks_and_pinv_cdf_cov(X)
        )
        return self

    def _evaluate(self, cuts: np.ndarray):
        """Evaluate the change score for a split within an interval.

        For each row in cuts, evaluates the score for [start, split] vs. [split, end].

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array with three columns: start, split, end.

        Returns
        -------
        scores : np.ndarray
            A 2D array of change scores, shape (n_cuts, 1).
        """
        scores = direct_rank_score(cuts, self._centered_data_ranks, self._pinv_rank_cov)
        return scores.reshape(-1, 1)

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate.

        Returns
        -------
        int
            The minimum valid size of an interval to evaluate.
        """
        return 2

    def get_model_size(self, p: int) -> int:
        """Get the number of parameters estimated by the score in each segment.

        Parameters
        ----------
        p : int
            Number of variables in the data.

        Returns
        -------
        int
            Number of parameters in the score function.
        """
        return p * (p + 1) // 2

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        params = [{}, {}]
        return params
