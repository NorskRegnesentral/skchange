"""CUSUM score."""

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted

from skchange.new_api.interval_scorers._base import BaseChangeScore
from skchange.new_api.typing import ArrayLike, Self
from skchange.new_api.utils.validation import validate_data
from skchange.penalties import make_bic_penalty
from skchange.utils.numba import njit
from skchange.utils.numba.stats import col_cumsum


@njit
def cusum_score(
    starts: np.ndarray,
    splits: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
) -> np.ndarray:
    """Calculate CUSUM score for change in mean at a split within intervals.

    Parameters
    ----------
    starts : np.ndarray
        Start indices for each interval.
    splits : np.ndarray
        Split indices for each interval.
    ends : np.ndarray
        End indices for each interval.
    sums : np.ndarray
        Cumulative sum of input data with an initial zero row.

    Returns
    -------
    np.ndarray
        Absolute weighted mean differences for each interval and feature.
    """
    n = ends - starts
    before_n = splits - starts
    after_n = ends - splits
    before_sum = sums[splits] - sums[starts]
    after_sum = sums[ends] - sums[splits]
    before_weight = np.sqrt(after_n / (n * before_n)).reshape(-1, 1)
    after_weight = np.sqrt(before_n / (n * after_n)).reshape(-1, 1)
    return np.abs(before_weight * before_sum - after_weight * after_sum)


class CUSUM(BaseChangeScore):
    """CUSUM change score for a change in mean.

    Computes the classical CUSUM statistic as the weighted absolute difference
    between the means before and after a split point within each interval.

    Notes
    -----
    Unlike `L2Cost`, this scorer has no fixed-vs-optim parameter variants,
    so no mode dispatch logic is required.
    """

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit CUSUM scorer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Used to validate shape and feature count.
        y : None
            Ignored.

        Returns
        -------
        self : CUSUM
            Fitted scorer.
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute cumulative sums for CUSUM evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            Dictionary with cumulative sums under key ``"sums"``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        return {"sums": col_cumsum(X, init_zero=True)}

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate CUSUM score at splits within intervals.

        Parameters
        ----------
        cache : dict
            Cache from precompute().
        interval_specs : array-like of shape (n_interval_specs, 3)
            Interval boundaries and split locations ``[start, split, end)``.

        Returns
        -------
        scores : ndarray of shape (n_interval_specs, n_features)
            CUSUM scores for each interval specification and feature.
        """
        check_is_fitted(self)
        sums = cache["sums"]

        interval_specs = check_array(
            interval_specs,
            ensure_2d=True,
            ensure_min_features=self.interval_specs_ncols,
        )
        if interval_specs.shape[1] != self.interval_specs_ncols:
            raise ValueError(
                f"interval_specs must have shape"
                f" (n_interval_specs, {self.interval_specs_ncols}), "
                f"got {interval_specs.shape}."
            )

        starts = interval_specs[:, 0]
        splits = interval_specs[:, 1]
        ends = interval_specs[:, 2]

        if np.any(splits <= starts) or np.any(splits >= ends):
            raise ValueError(
                "Each interval specification must satisfy start < split < end "
                "for CUSUM evaluation."
            )

        return cusum_score(starts, splits, ends, sums)

    def get_default_penalty(self) -> float:
        """Get default penalty value for the fitted CUSUM score."""
        bic_penalty = make_bic_penalty(self.n_features_in_, self.n_samples_in_)
        # BIC works on a squared error scale, while CUSUM is on an absolute error scale.
        return np.sqrt(bic_penalty)
