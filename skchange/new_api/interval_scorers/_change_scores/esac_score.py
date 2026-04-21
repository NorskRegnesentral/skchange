"""ESAC score for detecting changes in the mean of high-dimensional data."""

__author__ = ["peraugustmoen", "Tveten"]

from numbers import Real

import numpy as np
from scipy.stats import norm
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseChangeScore
from skchange.new_api.interval_scorers._change_scores.cusum import cusum_score
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._param_validation import Interval, _fit_context
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit
from skchange.utils.numba.stats import col_cumsum


@njit
def _transform_esac(
    cusum_scores: np.ndarray,
    a_s: np.ndarray,
    nu_s: np.ndarray,
    t_s: np.ndarray,
    threshold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute ESAC scores from CUSUM scores.

    Calculates the penalised score for the ESAC algorithm, as defined in
    Equation (6) in [1]_.

    Parameters
    ----------
    cusum_scores : np.ndarray
        A 2D array where each row represents the CUSUM scores for a specific
        candidate change location.
    a_s : np.ndarray
        A 1D array of hard threshold values. Correspond to ``a(t)`` as defined
        in Equation (4) in [1]_ for each ``t`` specified in ``t_s``.
    nu_s : np.ndarray
        A 1D array of mean-centering terms. Correspond to ``nu(t)`` as defined
        after Equation (4) in [1]_ for each ``t`` specified in ``t_s``.
    t_s : np.ndarray
        A 1D array of candidate sparsity values corresponding to the elements
        in ``a_s`` and ``nu_s``.
    threshold : np.ndarray
        A 1D array of penalty values, corresponding to ``gamma(t)`` in
        Equation (4) in [1]_, where ``t`` is as defined in ``t_s``.

    Returns
    -------
    output_scores : np.ndarray of shape (n_cuts, 1)
        Computed ESAC scores. Each element represents the maximum score for
        each candidate change location.
    sargmax : np.ndarray of shape (n_cuts,)
        Sparsity level at which the maximum score was achieved for each
        candidate change location.

    References
    ----------
    .. [1] Per August Jarval Moen, Ingrid Kristine Glad, Martin Tveten. Efficient
       sparsity adaptive changepoint estimation. Electron. J. Statist. 18 (2)
       3975 - 4038, 2024. https://doi.org/10.1214/24-EJS2294.
    """
    num_levels = len(threshold)
    num_cusum_scores = len(cusum_scores)
    output_scores = np.zeros(num_cusum_scores, dtype=np.float64)
    sargmax = np.zeros(num_cusum_scores, dtype=np.int64)

    for i in range(num_cusum_scores):
        temp_max = -np.inf
        for j in range(num_levels):
            temp_vec = (cusum_scores[i])[np.abs(cusum_scores[i]) > a_s[j]]
            if len(temp_vec) > 0:
                temp = np.sum(temp_vec**2 - nu_s[j]) - threshold[j]
                if temp > temp_max:
                    temp_max = temp
                    sargmax[i] = t_s[j]

        output_scores[i] = temp_max

    return output_scores.reshape(-1, 1), sargmax


class ESACScore(BaseChangeScore):
    """ESAC score for detecting changes in the mean of high-dimensional data.

    This is the sparsity adaptive penalised CUSUM score for a change in the mean.
    The ESAC score is a penalised version of the CUSUM score, where the CUSUM of
    each time series is thresholded, mean-centered and penalised by a
    sparsity-dependent penalty. The score is defined in Equation (6) in [1]_.

    Parameters
    ----------
    threshold_dense : float, default=1.5
        The leading constant in the penalty function taken as in (8) in [1]_ in
        the dense case where the candidate sparsity level ``t`` is greater than
        or equal to ``sqrt(p * log(n))``.
    threshold_sparse : float, default=1.0
        The leading constant in the penalty function taken as in (8) in [1]_ in
        the sparse case where the candidate sparsity level ``t`` is less than
        ``sqrt(p * log(n))``.

    Notes
    -----
    The ESAC score is inherently penalised (the internal thresholding and
    mean-centering make it self-penalised). The ``non_negative_scores`` tag is
    set to ``False`` because the score can be negative when no sparse signal
    exceeds the internal thresholds.

    References
    ----------
    .. [1] Per August Jarval Moen, Ingrid Kristine Glad, Martin Tveten. Efficient
       sparsity adaptive changepoint estimation. Electron. J. Statist. 18 (2)
       3975 - 4038, 2024. https://doi.org/10.1214/24-EJS2294.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import ESACScore
    >>> X = np.random.default_rng(0).normal(size=(100, 10))
    >>> scorer = ESACScore()
    >>> scorer.fit(X)
    ESACScore()
    >>> cache = scorer.precompute(X)
    >>> scorer.evaluate(cache, np.array([[0, 25, 50], [50, 75, 100]]))
    """

    _parameter_constraints: dict = {
        "threshold_dense": [Interval(Real, 0, None, closed="neither")],
        "threshold_sparse": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        threshold_dense: float = 1.5,
        threshold_sparse: float = 1.0,
    ):
        self.threshold_dense = threshold_dense
        self.threshold_sparse = threshold_sparse

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags: aggregated, penalised, and non_negative_scores=False."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.aggregated = True
        tags.interval_scorer_tags.penalised = True
        tags.interval_scorer_tags.non_negative_scores = False
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
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
            self._t_s_ = np.array([1])
            self._threshold_ = np.array(
                [self.threshold_dense * (np.sqrt(p * np.log(n)) + np.log(n))]
            )
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

            self._threshold_ = np.zeros_like(ss, dtype=float)
            self._threshold_[0] = self.threshold_dense * (
                np.sqrt(4 * p * np.log(n)) + 4 * np.log(n)
            )
            self._threshold_[1:] = self.threshold_sparse * (
                ss[1:] * np.log(np.exp(1) * p * 4 * np.log(n) / ss[1:] ** 2)
                + 4 * np.log(n)
            )

        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Store cumulative sums for segment-wise CUSUM evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=False)
        return {"sums": col_cumsum(X, init_zero=True)}

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate the ESAC score on intervals.

        Parameters
        ----------
        cache : dict
            Output from :meth:`precompute`.
        interval_specs : array-like of shape (n_intervals, 3)
            Each row is ``[start, split, end)``.

        Returns
        -------
        scores : ndarray of shape (n_intervals, 1)
            ESAC score for each interval.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(interval_specs, self.interval_specs_ncols)
        starts = interval_specs[:, 0]
        splits = interval_specs[:, 1]
        ends = interval_specs[:, 2]

        raw_cusum = cusum_score(starts, splits, ends, cache["sums"])
        scores, self.sargmaxes_ = _transform_esac(
            raw_cusum,
            self._a_s_,
            self._nu_s_,
            self._t_s_,
            self._threshold_,
        )
        return scores
