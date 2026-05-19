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
    a_s: np.ndarray,
    nu_s: np.ndarray,
    t_s: np.ndarray,
    gamma_s: np.ndarray,
) -> np.ndarray:
    r"""Compute unpenalised ESAC ratio scores from CUSUM scores.

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
    cusum_scores : np.ndarray of shape (n_cuts, n_features)
        CUSUM scores for each candidate split.
    a_s : np.ndarray of shape (n_levels,)
        Hard thresholds a(t) from Equation (4) in [1]_.
    nu_s : np.ndarray of shape (n_levels,)
        Mean-centering terms ν(t) from the paper.
    t_s : np.ndarray of shape (n_levels,)
        Candidate sparsity values (unused here, kept for API symmetry with the
        original ``_transform_esac``).
    gamma_s : np.ndarray of shape (n_levels,)
        Penalty values γ(t) with the leading constant C factored out (i.e.
        computed with C=1).  The calling code multiplies the result by C via
        the outer ``PenalisedScore`` penalty.

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
    num_levels = len(gamma_s)
    num_cuts = len(cusum_scores)
    output_scores = np.full(num_cuts, -np.inf, dtype=np.float64)

    for i in range(num_cuts):
        z = cusum_scores[i]
        for j in range(num_levels):
            mask = np.abs(z) > a_s[j]
            if not np.any(mask):
                continue
            raw = np.sum(z[mask] ** 2 - nu_s[j])
            ratio = raw / gamma_s[j]
            if ratio > output_scores[i]:
                output_scores[i] = ratio

    return output_scores.reshape(-1, 1)


class ESACScore(BaseChangeScore):
    """ESAC score for detecting changes in the mean of high-dimensional data.

    This is the sparsity-adaptive CUSUM score for a change in the mean,
    reformulated as a ratio statistic so that it can be composed with
    :class:`~skchange.new_api.interval_scorers.PenalisedScore`.

    For each candidate split the score is

    .. math::

        \\max_s \\frac{\\sum_{|z_j| > a_s} (z_j^2 - \\nu_s)}{\\gamma_s}

    where :math:`z` is the CUSUM statistic, and :math:`a_s`, :math:`\\nu_s`,
    :math:`\\gamma_s` are sparsity-dependent thresholds computed from the data
    dimensions as in [1]_.  A positive score indicates evidence for a change.

    The natural detection threshold is C = 1, which corresponds to the penalty
    value returned by :meth:`get_default_penalty`.  To use ESAC in a detector,
    wrap it in :class:`~skchange.new_api.interval_scorers.PenalisedScore`::

        PenalisedScore(ESACScore())               # uses default threshold C=1
        PenalisedScore(ESACScore(), penalty=C)    # custom threshold

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

    _parameter_constraints: dict = {}

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

        self._sums_ = col_cumsum(X, init_zero=True)

        if p == 1:
            self._a_s_ = np.array([0.0])
            self._nu_s_ = np.array([1.0])
            self._t_s_ = np.array([1], dtype=float)
            # gamma with C=1: (sqrt(p*log(n)) + log(n))
            self._gamma_s_ = np.array([np.sqrt(p * np.log(n)) + np.log(n)])
        else:
            max_s = min(np.sqrt(p * np.log(n)), p)
            log2ss = np.arange(0, np.floor(np.log2(max_s)) + 1)
            ss = 2**log2ss
            ss = np.concatenate(([p], ss[::-1]))
            self._t_s_ = np.array(ss, dtype=float)
            ss = np.array(ss, dtype=float)

            self._a_s_ = np.zeros_like(ss, dtype=float)
            self._a_s_[1:] = np.sqrt(
                2 * np.log(np.exp(1) * p * 4 * np.log(n) / ss[1:] ** 2)
            )

            log_dnorm = norm.logpdf(self._a_s_)
            log_pnorm_upper = norm.logsf(self._a_s_)
            self._nu_s_ = 1 + self._a_s_ * np.exp(log_dnorm - log_pnorm_upper)

            # gamma_s with leading constant C factored out (C=1 baked into
            # get_default_penalty; outer PenalisedScore multiplies by C).
            self._gamma_s_ = np.zeros_like(ss, dtype=float)
            self._gamma_s_[0] = np.sqrt(4 * p * np.log(n)) + 4 * np.log(n)
            self._gamma_s_[1:] = ss[1:] * np.log(
                np.exp(1) * p * 4 * np.log(n) / ss[1:] ** 2
            ) + 4 * np.log(n)

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
            self._a_s_,
            self._nu_s_,
            self._t_s_,
            self._gamma_s_,
        )
        return scores
