"""Empirical distribution function (EDF) saving for a fixed baseline CDF.

WORK IN PROGRESS. Not ready for use yet.
"""

__author__ = ["johannvk"]

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseSaving
from skchange.new_api.interval_scorers._costs.edf_cost import _cumulative_edf
from skchange.new_api.penalties import mvcapa_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._param_validation import _fit_context
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit
from skchange.utils.numba.general import compute_finite_difference_derivatives


@njit
def _edf_fixed_cdf_weights(
    fixed_quantiles: np.ndarray,
    quantile_points: np.ndarray,
) -> np.ndarray:
    """Compute quantile integration weights for the fixed CDF cost.

    Parameters
    ----------
    fixed_quantiles : np.ndarray
        CDF values at quantile points, shape (n_quantiles,).
    quantile_points : np.ndarray
        Data values at quantile points, shape (n_quantiles,).

    Returns
    -------
    weights : np.ndarray of shape (n_quantiles,)
        Integration weights for the fixed CDF cost.
    """
    if len(fixed_quantiles) < 3:
        raise ValueError("At least three fixed quantile values are required.")
    reciprocal_weights = 1.0 / (fixed_quantiles * (1.0 - fixed_quantiles))
    derivatives = compute_finite_difference_derivatives(
        ts=quantile_points, ys=fixed_quantiles
    )
    return reciprocal_weights * derivatives


@njit
def _edf_kl_saving(
    cumulative_edf: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    log_quantiles: np.ndarray,
    log_one_minus_quantiles: np.ndarray,
    weights: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the weighted KL divergence saving from a cumulative EDF cache.

    Computes ``2n * KL(segment_edf || baseline_cdf)`` weighted by the Fisher
    information measure of the baseline CDF.  Both the cross-entropy and
    self-entropy terms are evaluated on the *same* baseline quadrature grid,
    so the saving is exactly zero when the segment's empirical distribution
    matches the baseline.

    Parameters
    ----------
    cumulative_edf : np.ndarray of shape (n_samples + 1, n_quantiles)
        Cumulative EDF cache from ``_cumulative_edf`` evaluated at the
        baseline quantile points.
    starts : np.ndarray
        Start indices of segments (inclusive).
    ends : np.ndarray
        End indices of segments (exclusive).
    log_quantiles : np.ndarray of shape (n_quantiles,)
        Log of baseline CDF values at the quantile points.
    log_one_minus_quantiles : np.ndarray of shape (n_quantiles,)
        Log of one minus baseline CDF values at the quantile points.
    weights : np.ndarray of shape (n_quantiles,)
        Fisher-information integration weights from ``_edf_fixed_cdf_weights``.
    out : np.ndarray or None
        Output array of shape (n_segments,). Created if None.

    Returns
    -------
    out : np.ndarray of shape (n_segments,)
        Weighted KL divergence saving for each segment.
    """
    n_quantiles = cumulative_edf.shape[1]
    segment_edf = np.zeros(n_quantiles, dtype=np.float64)
    if out is None:
        out = np.zeros(len(starts), dtype=np.float64)
    for i, (start, end) in enumerate(zip(starts, ends)):
        n = end - start
        segment_edf[:] = (cumulative_edf[end, :] - cumulative_edf[start, :]) / float(n)
        kl = 0.0
        for j in range(n_quantiles):
            f = segment_edf[j]
            w = weights[j]
            if f > 1e-10:
                kl += f * (np.log(f) - log_quantiles[j]) * w
            if f < 1.0 - 1e-10:
                kl += (1.0 - f) * (np.log(1.0 - f) - log_one_minus_quantiles[j]) * w
        out[i] = 2.0 * n * kl
    return out


class EDFSaving(BaseSaving):
    """Empirical distribution function (EDF) saving for a fixed baseline CDF.

    Measures the reduction in EDF cost when fitting the empirical CDF of a
    segment compared to a fixed reference CDF. A large saving indicates that
    the baseline CDF is a poor fit for the data in that segment.

    The saving equals twice the integrated KL divergence of the segment's
    empirical CDF from the baseline CDF, approximated using the baseline
    quadrature grid::

        saving(s, e) = 2 * (e - s) * sum_j [
            F_n(t_j) * log(F_n(t_j) / G(t_j))
            + (1 - F_n(t_j)) * log((1 - F_n(t_j)) / (1 - G(t_j)))
        ] * w_j

    where ``t_j`` are the baseline quantile points, ``G(t_j)`` are the
    baseline CDF values, ``F_n(t_j)`` is the segment's empirical CDF, and
    ``w_j = dG/dt / (G*(1-G))`` are Fisher-information integration weights.
    Both terms use the same quadrature grid, so the saving is zero when the
    segment's distribution matches the baseline [1]_.

    Parameters
    ----------
    baseline_quantile_points : array-like of shape (n_quantiles, n_features) \
            or (n_quantiles,)
        Sorted sample values at which the baseline CDF is evaluated.
    baseline_quantile_values : array-like of shape (n_quantiles, n_features) \
            or (n_quantiles,)
        CDF values at the corresponding ``baseline_quantile_points``. Must be
        strictly increasing and in ``(0, 1)``.

    Attributes
    ----------
    baseline_quantile_points_ : np.ndarray of shape (n_quantiles, n_features)
        Validated baseline quantile points.
    baseline_quantile_values_ : np.ndarray of shape (n_quantiles, n_features)
        Validated and clipped baseline quantile values.

    Notes
    -----
    Requires at least ``n_quantiles`` (baseline) observations per segment.

    References
    ----------
    .. [1] Haynes, K., Fearnhead, P. & Eckley, I.A. A computationally efficient
       nonparametric approach for changepoint detection. Stat Comput 27, 1293-1305
       (2017). https://doi.org/10.1007/s11222-016-9687-5

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import EDFSaving
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 2))
    >>> quantile_probs = np.linspace(0.05, 0.95, 20)
    >>> qpoints = np.quantile(X, quantile_probs, axis=0)
    >>> qvalues = np.tile(quantile_probs[:, None], (1, 2))
    >>> scorer = EDFSaving(qpoints, qvalues)
    >>> scorer.fit(X)
    EDFSaving(...)
    >>> cache = scorer.precompute(X)
    >>> interval_specs = np.array([[0, 50], [50, 100]])
    >>> scorer.evaluate(cache, interval_specs)
    """

    _parameter_constraints: dict = {
        "baseline_quantile_points": ["array-like"],
        "baseline_quantile_values": ["array-like"],
    }

    def __init__(
        self,
        baseline_quantile_points: ArrayLike,
        baseline_quantile_values: ArrayLike,
    ):
        self.baseline_quantile_points = baseline_quantile_points
        self.baseline_quantile_values = baseline_quantile_values

    @property
    def min_size(self) -> int:
        """Minimum segment size: the number of baseline quantiles."""
        check_is_fitted(self)
        return self.baseline_quantile_points_.shape[0]

    def get_default_penalty(self) -> np.ndarray:
        """Get the default penalty for the fitted EDF saving.

        Uses the number of baseline quantiles as ``n_params_per_feature`` to
        account for the non-parametric nature of the EDF saving.

        Returns
        -------
        np.ndarray of shape (n_features,)
            Default penalty value for each number of affected features.
        """
        check_is_fitted(self)
        n_quantiles = self.baseline_quantile_points_.shape[0]
        return mvcapa_penalty(
            self.n_samples_in_, self.n_features_in_, n_params_per_feature=n_quantiles
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit the EDF saving, validating the baseline CDF.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Used only to determine the number of features.
        y : None
            Ignored.

        Returns
        -------
        self : EDFSaving
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)
        n_samples, n_features = X.shape

        # Validate and broadcast baseline params
        qpoints = np.asarray(self.baseline_quantile_points, dtype=np.float64)
        qvalues = np.asarray(self.baseline_quantile_values, dtype=np.float64)

        if qpoints.shape != qvalues.shape:
            raise ValueError(
                "baseline_quantile_points and baseline_quantile_values must "
                f"have the same shape, got {qpoints.shape} and {qvalues.shape}."
            )
        if qpoints.ndim > 2:
            raise ValueError(
                f"baseline_quantile_points must be 1D or 2D, got shape {qpoints.shape}."
            )
        if qpoints.ndim == 1:
            qpoints = qpoints.reshape(-1, 1)
            qvalues = qvalues.reshape(-1, 1)

        if not np.all(np.diff(qpoints, axis=0) > 0):
            raise ValueError(
                "baseline_quantile_points must be strictly increasing along axis 0."
            )
        if not np.all(np.diff(qvalues, axis=0) > 0):
            raise ValueError(
                "baseline_quantile_values must be strictly increasing along axis 0."
            )
        if not (np.all(qvalues >= 0) and np.all(qvalues <= 1)):
            raise ValueError(
                "baseline_quantile_values must be in the closed interval [0, 1]."
            )

        # Broadcast single-column baseline to all features
        if qpoints.shape[1] == 1 and n_features > 1:
            qpoints = np.tile(qpoints, (1, n_features))
            qvalues = np.tile(qvalues, (1, n_features))

        if qpoints.shape[1] != n_features:
            raise ValueError(
                f"baseline_quantile_points has {qpoints.shape[1]} feature columns "
                f"but X has {n_features} features."
            )

        # Clip quantile values to avoid log(0)
        qvalues = np.clip(qvalues, 1e-10, 1 - 1e-10)

        self.baseline_quantile_points_ = qpoints
        self.baseline_quantile_values_ = qvalues
        self.log_fixed_quantiles_ = np.log(qvalues)
        self.log_one_minus_fixed_quantiles_ = np.log(1.0 - qvalues)

        n_quantiles = qpoints.shape[0]
        self.quantile_weights_ = np.zeros((n_quantiles, n_features))
        for col in range(n_features):
            self.quantile_weights_[:, col] = _edf_fixed_cdf_weights(
                qvalues[:, col], qpoints[:, col]
            )
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Build per-feature EDF caches at the baseline quantile points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            Dictionary with key:

            ``"edf_caches"`` : list of length n_features
                Cumulative EDF evaluated at baseline quantile points.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        n_features = X.shape[1]
        edf_caches = [
            _cumulative_edf(X[:, col], self.baseline_quantile_points_[:, col])
            for col in range(n_features)
        ]
        return {"edf_caches": edf_caches}

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate EDF saving on intervals.

        Parameters
        ----------
        cache : dict
            Cache from precompute().
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        savings : ndarray of shape (n_interval_specs, n_features)
            EDF savings for each interval and feature.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        n_intervals = len(starts)
        n_features = len(cache["edf_caches"])

        savings = np.zeros((n_intervals, n_features))

        for col in range(n_features):
            _edf_kl_saving(
                cumulative_edf=cache["edf_caches"][col],
                starts=starts,
                ends=ends,
                log_quantiles=self.log_fixed_quantiles_[:, col],
                log_one_minus_quantiles=self.log_one_minus_fixed_quantiles_[:, col],
                weights=self.quantile_weights_[:, col],
                out=savings[:, col],
            )

        return savings
