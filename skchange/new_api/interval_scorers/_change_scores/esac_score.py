"""ESAC score for detecting changes in the mean of high-dimensional data."""

__author__ = ["peraugustmoen", "Tveten"]

import numpy as np
from scipy.stats import norm
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseChangeScore
from skchange.new_api.interval_scorers._change_scores.cusum import cusum_score
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit
from skchange.utils.numba.stats import col_cumsum


@njit
def _transform_esac_ratio(
    cusum_scores: np.ndarray,
    coordinate_thresholds: np.ndarray,
    mean_corrections: np.ndarray,
    sparsity_levels: np.ndarray,
    sparsity_penalties: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute ESAC scores from CUSUM scores.

    For each candidate change location i and each sparsity level j, computes

        raw_j(i) = sum(z² - ν_j  for  z in cusum_scores[i]  with |z| > a_j)
        ratio_j(i) = raw_j(i) / gamma_j

    and returns score[i] = max_j ratio_j(i).

    When no component exceeds the hard threshold a_j for any j, the score is
    ``-inf`` (no signal found).  A positive score indicates that the raw ESAC
    statistic exceeds the penalty ``gamma_j`` for some j, mirroring the
    detection rule of the original penalised formulation with threshold C=1.

    Parameters
    ----------
    cusum_scores : np.ndarray
        A 2D array where each row represents the CUSUM scores for a specific
        candidate change location.
    coordinate_thresholds : np.ndarray
        A 1D array of hard threshold values. Correspond to ``a(t)`` as defined
        in Equation (4) in [1]_ for each sparsity ``t`` in ``sparsity_levels``.
    mean_corrections : np.ndarray
        A 1D array of mean-centering terms. Correspond to ``nu(t)`` as defined
        after Equation (4) in [1]_ for each sparsity ``t`` in
        ``sparsity_levels``.
    sparsity_levels : np.ndarray
        A 1D array of candidate sparsity values corresponding to the elements
        in ``coordinate_thresholds`` and ``mean_corrections``.
    sparsity_penalties : np.ndarray
        A 1D array of penalty values, corresponding to ``gamma(t)`` in
        Equation (4) in [1]_, where ``t`` is as defined in ``sparsity_levels``.

    Returns
    -------
    output_scores : np.ndarray of shape (n_cuts, 1)
        ESAC ratio score for each candidate split: max_j raw_j / gamma_j.

    References
    ----------
    .. [1] Per August Jarval Moen, Ingrid Kristine Glad, Martin Tveten.
       Efficient sparsity adaptive changepoint estimation. Electron. J. Statist.
       18 (2) 3975 - 4038, 2024. https://doi.org/10.1214/24-EJS2294.
    """
    num_levels = len(sparsity_penalties)
    num_cusum_scores = len(cusum_scores)
    output_scores = np.zeros(num_cusum_scores, dtype=np.float64)
    sargmax = np.zeros(num_cusum_scores, dtype=np.int64)

    for i in range(num_cuts):
        z = cusum_scores[i]
        for j in range(num_levels):
            temp_vec = (cusum_scores[i])[
                np.abs(cusum_scores[i]) > coordinate_thresholds[j]
            ]
            if len(temp_vec) > 0:
                temp = np.sum(temp_vec**2 - mean_corrections[j]) - sparsity_penalties[j]
                if temp > temp_max:
                    temp_max = temp
                    sargmax[i] = sparsity_levels[j]

    return output_scores.reshape(-1, 1)


class ESACScore(BaseChangeScore):
    """ESAC score for detecting changes in the mean of high-dimensional data.

    This is the sparsity-adaptive CUSUM score for a change in the mean,
    reformulated as a ratio statistic so that it can be composed with
    :class:`~skchange.new_api.interval_scorers.PenalisedScore`.

    Parameters
    ----------
    penalty_scale_dense : float, default=2.0
        The leading constant in the penalty function taken as in (8) in [1]_ in
        the dense case where the candidate sparsity level ``t`` is greater than
        or equal to ``sqrt(p * log(n))``.
    penalty_scale_sparse : float, default=1.5
        The leading constant in the penalty function taken as in (8) in [1]_ in
        the sparse case where the candidate sparsity level ``t`` is less than
        ``sqrt(p * log(n))``.

    Attributes
    ----------
    sparsity_levels_ : np.ndarray
        Candidate sparsity values ``t`` (denoted ``t_s`` in [1]_).
    coordinate_thresholds_ : np.ndarray
        Per-coordinate hard thresholds ``a(t)`` applied to each CUSUM entry,
        as defined in Equation (4) in [1]_.
    mean_corrections_ : np.ndarray
        Mean-centering terms ``nu(t)`` subtracted from the squared thresholded
        CUSUMs, as defined after Equation (4) in [1]_.
    sparsity_penalties_ : np.ndarray
        Per-sparsity penalty values ``gamma(t)`` subtracted in the final
        score, as defined in Equation (4) in [1]_.

    Notes
    -----
    The ``non_negative_scores`` tag is ``False`` because the ratio score is
    negative when no component exceeds the hard threshold.

    References
    ----------
    .. [1] Per August Jarval Moen, Ingrid Kristine Glad, Martin Tveten. Efficient
       sparsity adaptive changepoint estimation. Electron. J. Statist. 18 (2)
       3975 - 4038, 2024. https://doi.org/10.1214/24-EJS2294.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import ESACScore, PenalisedScore
    >>> X = np.random.default_rng(0).normal(size=(100, 10))
    >>> scorer = PenalisedScore(ESACScore())
    >>> scorer.fit(X)
    PenalisedScore(scorer=ESACScore())
    >>> cache = scorer.precompute(X)
    >>> scorer.evaluate(cache, np.array([[0, 25, 50], [50, 75, 100]]))
    """

    _parameter_constraints: dict = {
        "penalty_scale_dense": [Interval(Real, 0, None, closed="neither")],
        "penalty_scale_sparse": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        penalty_scale_dense: float = 2.0,
        penalty_scale_sparse: float = 1.5,
    ):
        self.penalty_scale_dense = penalty_scale_dense
        self.penalty_scale_sparse = penalty_scale_sparse

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags: aggregated, unpenalised, non_negative_scores=False."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.aggregated = True
        tags.interval_scorer_tags.penalised = False
        tags.interval_scorer_tags.non_negative_scores = False
        return tags

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit the score by computing ESAC thresholds from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : ESACScore
        """
        X = validate_data(
            self, X, ensure_2d=True, dtype=np.float64, reset=True, ensure_min_samples=2
        )
        n, p = X.shape

        if p == 1:
            self.coordinate_thresholds_ = np.array([0.0])
            self.mean_corrections_ = np.array([1.0])
            self.sparsity_levels_ = np.array([1])
            self.sparsity_penalties_ = np.array(
                [self.penalty_scale_dense * (np.sqrt(p * np.log(n)) + np.log(n))]
            )
        else:
            max_s = min(np.sqrt(p * np.log(n)), p)
            log2ss = np.arange(0, np.floor(np.log2(max_s)) + 1)
            ss = 2**log2ss
            ss = np.concatenate(([p], ss[::-1]))
            self.sparsity_levels_ = np.array(ss, dtype=float)
            ss = np.array(ss, dtype=float)

            self.coordinate_thresholds_ = np.zeros_like(ss, dtype=float)
            self.coordinate_thresholds_[1:] = np.sqrt(
                2 * np.log(np.exp(1) * p * 4 * np.log(n) / ss[1:] ** 2)
            )

            log_dnorm = norm.logpdf(self.coordinate_thresholds_)
            log_pnorm_upper = norm.logsf(self.coordinate_thresholds_)
            self.mean_corrections_ = 1 + self.coordinate_thresholds_ * np.exp(
                log_dnorm - log_pnorm_upper
            )

            self.sparsity_penalties_ = np.zeros_like(ss, dtype=float)
            self.sparsity_penalties_[0] = self.penalty_scale_dense * (
                np.sqrt(4 * p * np.log(n)) + 4 * np.log(n)
            )
            self.sparsity_penalties_[1:] = self.penalty_scale_sparse * (
                ss[1:] * np.log(np.exp(1) * p * 4 * np.log(n) / ss[1:] ** 2)
                + 4 * np.log(n)
            )

        return self

    def get_default_penalty(self) -> float:
        """Return the default penalty threshold C = 1.0.

        The theoretical threshold for the ESAC ratio statistic is C = 1.  At
        this value the type-I error is controlled asymptotically as in [1]_.

        Returns
        -------
        float
            1.0
        """
        check_is_fitted(self)
        return 1.0

    def precompute(self, X: ArrayLike) -> dict:
        """Store cumulative sums for segment-wise CUSUM evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            Dictionary with key ``"sums"`` (cumulative column sums).
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=False)
        return {"sums": col_cumsum(X, init_zero=True)}

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate the ESAC ratio score on intervals.

        Parameters
        ----------
        cache : dict
            Output from :meth:`precompute`.
        interval_specs : array-like of shape (n_intervals, 3)
            Each row is ``[start, split, end)``.

        Returns
        -------
        scores : ndarray of shape (n_intervals, 1)
            ESAC ratio score for each interval.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(interval_specs, self.interval_specs_ncols)
        starts = interval_specs[:, 0]
        splits = interval_specs[:, 1]
        ends = interval_specs[:, 2]

        raw_cusum = cusum_score(starts, splits, ends, cache["sums"])
        scores = _transform_esac_ratio(
            raw_cusum,
            self.coordinate_thresholds_,
            self.mean_corrections_,
            self.sparsity_levels_,
            self.sparsity_penalties_,
        )
        return scores
