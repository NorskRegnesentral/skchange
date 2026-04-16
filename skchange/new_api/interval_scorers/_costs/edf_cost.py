"""Empirical distribution function (EDF) cost function."""

__author__ = ["johannvk"]

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseCost
from skchange.new_api.penalties import bic_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit
from skchange.utils.numba.stats import col_cumsum


def _edf_quantile_points(
    X: np.ndarray, n_quantiles: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute quantile points for approximating the EDF integral.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (n_samples, n_features).
    n_quantiles : int
        Number of quantiles to use.

    Returns
    -------
    quantile_points : np.ndarray of shape (n_quantiles, n_features)
        Data values at each quantile for each feature.
    quantile_values : np.ndarray of shape (n_quantiles, n_features)
        Quantile probability values (same across feature columns).
    """
    n_samples = X.shape[0]
    edf_scaling = -np.log(2 * n_samples - 1)
    quantile_range = np.arange(1, n_quantiles + 1)
    integration_quantiles = 1.0 / (
        1.0 + np.exp(edf_scaling * ((2 * quantile_range - 1) / n_quantiles - 1))
    )
    quantile_points = np.quantile(X, integration_quantiles, axis=0)
    quantile_values = np.tile(integration_quantiles.reshape(-1, 1), (1, X.shape[1]))
    return quantile_points, quantile_values


@njit
def _cumulative_edf(xs: np.ndarray, quantile_points: np.ndarray) -> np.ndarray:
    """Build a cumulative EDF evaluated at fixed quantile points.

    Parameters
    ----------
    xs : np.ndarray
        1D data array of shape (n_samples,).
    quantile_points : np.ndarray
        Points at which to evaluate the EDF, shape (n_quantiles,).

    Returns
    -------
    cumulative_edf : np.ndarray of shape (n_samples + 1, n_quantiles)
        Cumulative EDF counts with a leading row of zeros.
    """
    lte_mask = (xs[:, None] < quantile_points[None, :]).astype(np.float64)
    lte_mask += 0.5 * (xs[:, None] == quantile_points[None, :])
    return col_cumsum(lte_mask, init_zero=True)


@njit(fastmath=True)
def _edf_mle_cost(
    cumulative_edf: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the approximate MLE EDF cost from a cumulative EDF cache.

    Parameters
    ----------
    cumulative_edf : np.ndarray of shape (n_samples + 1, n_quantiles)
        Cumulative EDF cache from ``_cumulative_edf``.
    starts : np.ndarray
        Start indices of segments (inclusive).
    ends : np.ndarray
        End indices of segments (exclusive).
    out : np.ndarray or None
        Output array of shape (n_segments,). Created if None.

    Returns
    -------
    out : np.ndarray of shape (n_segments,)
        MLE EDF cost for each segment.
    """
    n_cache, n_quantiles = cumulative_edf.shape
    segment_edf = np.zeros(n_quantiles, dtype=np.float64)
    if out is None:
        out = np.zeros(len(starts), dtype=np.float64)
    edf_scale = -np.log(2 * n_cache - 1)
    for i, (start, end) in enumerate(zip(starts, ends)):
        n = end - start
        segment_edf[:] = (cumulative_edf[end, :] - cumulative_edf[start, :]) / float(n)
        ll = 0.0
        for q in segment_edf:
            if q > 1e-10:
                ll += q * np.log(q)
            one_minus_q = 1.0 - q
            if one_minus_q > 1e-10:
                ll += one_minus_q * np.log(one_minus_q)
        ll *= (-2.0 * edf_scale / n_quantiles) * n
        out[i] = -2.0 * ll
    return out


class EDFCost(BaseCost):
    r"""Empirical distribution function (EDF) cost.

    Computes a non-parametric cost based on the approximate integrated
    log-likelihood of the empirical CDF for each segment. The cost is
    estimated using a quantile-based approximation of the EDF integral [1]_.

    .. math::
        C(X_{s:e}) \\approx -2 \\cdot \\frac{c}{Q} \\sum_{q=1}^{Q}
        \\hat{F}_n(t_q) \\log \\hat{F}_n(t_q)
        + (1 - \\hat{F}_n(t_q)) \\log(1 - \\hat{F}_n(t_q))

    where :math:`\\hat{F}_n` is the empirical CDF of the segment,
    :math:`t_q` are quantile points derived from the training data, and
    :math:`c` is a scaling constant.

    Parameters
    ----------
    n_approximation_quantiles : int or None, default=None
        Number of quantiles used to approximate the EDF integral. If ``None``,
        defaults to ``ceil(4 * log(n_samples))`` at fit time.

    Attributes
    ----------
    n_quantiles_ : int
        Number of quantiles used after fitting.
    quantile_points_ : np.ndarray of shape (n_quantiles_, n_features)
        Quantile point values for each feature, derived from training data.

    Notes
    -----
    Requires at least ``n_quantiles_`` observations per segment. The
    ``min_size`` property reflects this after fitting.

    References
    ----------
    .. [1] Haynes, K., Fearnhead, P. & Eckley, I.A. A computationally efficient
       nonparametric approach for changepoint detection. Stat Comput 27, 1293-1305
       (2017). https://doi.org/10.1007/s11222-016-9687-5

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import EDFCost
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> cost = EDFCost()
    >>> cost.fit(X)
    EDFCost()
    >>> cache = cost.precompute(X)
    >>> interval_specs = np.array([[0, 50], [50, 100]])
    >>> cost.evaluate(cache, interval_specs)
    """

    def __init__(self, n_approximation_quantiles: int | None = None):
        self.n_approximation_quantiles = n_approximation_quantiles

    @property
    def min_size(self) -> int:
        """Minimum segment size: the number of quantiles used."""
        check_is_fitted(self)
        return self.n_quantiles_

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit the EDF cost, computing quantile points from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : EDFCost
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)
        n_samples = X.shape[0]

        if self.n_approximation_quantiles is None:
            self.n_quantiles_ = int(np.ceil(4 * np.log(n_samples)))
        else:
            self.n_quantiles_ = self.n_approximation_quantiles

        self.quantile_points_, _ = _edf_quantile_points(X, self.n_quantiles_)
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Build per-feature cumulative EDF caches at training quantile points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            Dictionary with key ``"edf_caches"``: a list of length ``n_features``,
            each entry being a cumulative EDF array of shape
            ``(n_samples + 1, n_quantiles_)``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        edf_caches = [
            _cumulative_edf(X[:, col], self.quantile_points_[:, col])
            for col in range(X.shape[1])
        ]
        return {"edf_caches": edf_caches}

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate EDF cost on intervals.

        Parameters
        ----------
        cache : dict
            Cache from precompute().
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        costs : ndarray of shape (n_interval_specs, n_features)
            EDF costs for each interval and feature.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        edf_caches = cache["edf_caches"]
        n_intervals = len(starts)
        n_features = len(edf_caches)
        costs = np.zeros((n_intervals, n_features))
        for col in range(n_features):
            _edf_mle_cost(edf_caches[col], starts, ends, out=costs[:, col])
        return costs

    def get_default_penalty(self) -> float:
        r"""Get the default penalty for the fitted EDF cost.

        The two-sample EDF change score converges asymptotically to a
        functional of a squared Brownian bridge
        (:math:`\\int_0^1 B^2(u) \\, du`), analogous to an Anderson-Darling
        statistic. This distribution is heavier-tailed than any
        fixed-degree-of-freedom chi-square, so a plain BIC or chi-square
        penalty consistently underpenalises in finite samples.

        The penalty is set to ``2.5 * bic_penalty(n, 2 * p)`` where the
        factor 2.5 compensates for the heavier tail, and 2 parameters per
        feature reflects sensitivity to both location and scale shifts.

        Returns
        -------
        float
            Default penalty value.
        """
        check_is_fitted(self)
        return 2.5 * bic_penalty(self.n_samples_in_, 2 * self.n_features_in_)
