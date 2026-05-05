"""Utility functions for savings module."""

from typing import TYPE_CHECKING

import numpy as np

from skchange.new_api.typing import ArrayLike

if TYPE_CHECKING:
    pass


def resolve_baseline_location(
    param_value: ArrayLike | float | None,
    X: np.ndarray,
    param_name: str = "baseline_param",
) -> np.ndarray:
    """Resolve a baseline parameter to shape (n_features,) using median when None.

    This utility handles baseline parameters that use column-wise median as the
    default robust estimator when the parameter is None. It accepts scalar inputs
    (which are broadcasted to (n_features,)) or explicit array inputs.

    Parameters
    ----------
    param_value : array-like, float, or None
        User-supplied baseline parameter. If ``None``, the column-wise median
        of ``X`` is used as a robust estimate. If scalar, broadcasted to
        (n_features,). If array, must have shape (n_features,).
    X : np.ndarray of shape (n_samples, n_features)
        Fitted training data (already validated).
    param_name : str, default="baseline_param"
        Name of the parameter used in error messages.

    Returns
    -------
    resolved : np.ndarray of shape (n_features,)
        The resolved parameter broadcasted to (n_features,).

    Raises
    ------
    ValueError
        If param_value has incorrect shape.
    """
    n_features = X.shape[1]
    if param_value is None:
        return np.median(X, axis=0)
    resolved = np.asarray(param_value, dtype=np.float64)
    if resolved.ndim == 0:
        resolved = np.full(n_features, resolved)
    if resolved.shape != (n_features,):
        raise ValueError(
            f"{param_name} must be a scalar or array of shape "
            f"(n_features,)={(n_features,)}, got shape {resolved.shape}."
        )
    return resolved


def resolve_baseline_location_and_scatter(
    baseline_mean: "ArrayLike | float | None",
    baseline_scatter: "ArrayLike | float | None",
    X: np.ndarray,
    mean_param_name: str = "baseline_mean",
    scatter_param_name: str = "baseline_scatter",
) -> "tuple[np.ndarray, np.ndarray]":
    """Resolve a (mean, scatter) baseline pair for multivariate savings.

    Three cases are handled:

    * **Both** ``None`` — uses Minimum Covariance Determinant (MCD) for a
      joint robust estimate of location and scatter that is resistant to an
      outlier segment inside the training window.
    * **Mean given, scatter** ``None`` — mean is resolved via
      :func:`resolve_baseline_location`; scatter is estimated as the biased sample
      scatter matrix ``(X - mean).T @ (X - mean) / n``.
    * **Scatter given** — mean is resolved via :func:`resolve_baseline_location`;
      scatter is broadcast from a scalar to ``scalar * I`` or validated as a
      ``(n_features, n_features)`` array.

    Parameters
    ----------
    baseline_mean : array-like, float, or None
        Baseline location. Passed to :func:`resolve_baseline_location`.
    baseline_scatter : array-like, float, or None
        Baseline SPD matrix. A scalar is broadcast to ``scalar * I``.
        When ``None``, the scatter is estimated from data.
    X : np.ndarray of shape (n_samples, n_features)
        Fitted training data (already validated, dtype float64).
    mean_param_name : str, default="baseline_mean"
        Parameter name used in error messages for the mean.
    scatter_param_name : str, default="baseline_scatter"
        Parameter name used in error messages for the scatter matrix.

    Returns
    -------
    mean : np.ndarray of shape (n_features,)
    scatter : np.ndarray of shape (n_features, n_features)

    Raises
    ------
    ValueError
        If the data has fewer samples than features, making scatter estimation
        impossible.
    """
    n, p = X.shape

    if baseline_mean is None and baseline_scatter is None:
        if n <= p:
            raise ValueError(
                f"Cannot estimate a {p}x{p} scatter matrix from n_samples={n}. "
                f"Provide at least {p + 1} samples, or supply {mean_param_name} "
                f"and {scatter_param_name} explicitly."
            )
        from sklearn.covariance import MinCovDet

        mcd = MinCovDet(store_precision=True, assume_centered=False)
        mcd.fit(X)
        return mcd.location_, mcd.covariance_

    mean = resolve_baseline_location(baseline_mean, X, param_name=mean_param_name)

    if baseline_scatter is None:
        if n <= p:
            raise ValueError(
                f"Cannot estimate a {p}x{p} scatter matrix from n_samples={n}. "
                f"Provide at least {p + 1} samples, or supply "
                f"{scatter_param_name} explicitly."
            )
        centered = X - mean
        scatter = (centered.T @ centered) / n
    else:
        resolved = np.asarray(baseline_scatter, dtype=np.float64)
        if resolved.ndim == 0:
            scalar = float(resolved)
            if scalar <= 0:
                raise ValueError(f"{scatter_param_name} must be strictly positive.")
            scatter = scalar * np.eye(p, dtype=np.float64)
        else:
            if resolved.shape != (p, p):
                raise ValueError(
                    f"{scatter_param_name} must be a scalar or array of shape "
                    f"(n_features, n_features)={(p, p)}, got shape {resolved.shape}."
                )
            scatter = resolved

    return mean, scatter
