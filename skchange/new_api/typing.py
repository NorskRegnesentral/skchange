"""Typing definitions for the new skchange API.

Design Philosophy
-----------------
Input Convention:

- Single series: ArrayLike of shape (n_samples, n_features)
- Univariate series use n_features=1 (never 1D arrays)
- For processing multiple series, use loops or utilities external to the detector

This keeps the detector API simple and fully sklearn-compatible (pipelines,
GridSearchCV, etc.). Multi-series workflows are handled via GroupKFold or
custom loops that maintain user control over memory and parallelization.
"""

from __future__ import annotations

try:
    from typing import NotRequired, Self  # Python 3.11+
except ImportError:
    from typing_extensions import NotRequired, Self  # Python 3.8-3.10

from typing import Any, Protocol, TypedDict, runtime_checkable

import numpy as np

# Type alias for array-like inputs
# We use Any instead of numpy.typing.ArrayLike to avoid linter warnings about
# potentially compatible array types (e.g., pandas DataFrames, polars, xarray)
# that aren't covered by numpy's definition.
# Convention: ArrayLike is for INPUTS (flexible), np.ndarray for OUTPUTS (consistent).
ArrayLike = Any


class Segmentation(TypedDict):
    """Sparse segmentation representation for changepoint detection.

    This is the universal sparse format used throughout the API for both
    input (y labels) and output (predict results).

    TypedDict allows plain dicts to be returned, avoiding custom classes and
    maintaining alignment with scikit-learn's design philosophy.

    The result contains a sparse representation with per-changepoint arrays
    (changepoints, changed_features), per-segment arrays (labels),
    and scalar metadata (n_samples).

    Required Fields
    ---------------
    changepoints : np.ndarray of shape (n_changepoints,)
        Changepoint indices where the time series exhibits structural breaks.
        Integer array with values in [0, n_samples). Each index marks the start
        of a new segment. Empty array if no changepoints detected.
        dtype: typically int64, but any integer dtype accepted.

    labels : np.ndarray of shape (n_changepoints + 1,)
        Segment labels for each interval between changepoints (and boundaries).
        Integer array starting from 0. Default is [0, 1, 2, ...] where each
        segment has a unique label. Can assign the same label to multiple
        segments to group them into the same regime/state.
        dtype: typically int64, but any integer dtype accepted.

    n_samples : int
        Number of samples (timepoints) in the analyzed time series.
        Required for sparse-to-dense conversion and validation.
        Must be positive integer.

    Optional Fields
    ---------------
    n_features : int
        Number of features (variables/channels) in the analyzed time series.
        Must be positive integer, use 1 for univariate data.
        Useful metadata but not required for core sparse representation.

    changed_features : list[np.ndarray]
        Feature/channel indices affected at each changepoint.
        List of length n_changepoints. Each element is an integer array of
        shape (n_affected_i,) containing indices in [0, n_features).
        None or omit entirely if not applicable (e.g., univariate data) or
        if all features are affected at all changepoints.
        dtype per array: typically int64, but any integer dtype accepted.

    meta : dict[str, Any]
        Optional metadata about the detection process or results.
        May include algorithm parameters, thresholds, convergence info,
        computation time, or any detector-specific information.
        Structure is detector-dependent and not standardized.

    Examples
    --------
    >>> # Minimal result - only required fields
    >>> result = {
    ...     "changepoints": np.array([10, 50, 90]),
    ...     "labels": np.array([0, 1, 2, 3]),
    ...     "n_samples": 200,
    ... }

    >>> # Full result - all fields included
    >>> result = make_segmentation(
    ...     changepoints=np.array([10, 50, 90]),
    ...     labels=np.array([0, 1, 2, 3]),
    ...     n_samples=200,
    ...     n_features=1,
    ... )

    >>> # Auto-generated labels (recommended)
    >>> result = make_segmentation(
    ...     changepoints=np.array([10, 50]),
    ...     n_samples=100,
    ... )
    >>> result["labels"]  # Auto-generated: [0, 1, 2]
    array([0, 1, 2])

    >>> # Access fields via dict keys
    >>> cps = result["changepoints"]
    >>> labels = result["labels"]
    """

    changepoints: np.ndarray
    labels: np.ndarray
    n_samples: int
    n_features: NotRequired[int]
    changed_features: NotRequired[list[np.ndarray]]
    meta: NotRequired[dict[str, Any]]


@runtime_checkable
class ChangeDetector(Protocol):
    """Protocol for changepoint detection algorithms.

    Defines the minimal public interface for changepoint detectors.
    Only fit() and predict() are required.

    Terminology
    -----------
    **Univariate vs Multivariate**:

    - Univariate: n_features = 1 (single channel/variable)
    - Multivariate: n_features > 1 (multiple channels/variables)

    Design Principles
    -----------------
    **Single Series API**
        Both fit() and predict() operate on single time series. For workflows
        involving multiple series (e.g., cross-series hyperparameter tuning),
        use external tools like sklearn's GroupKFold or custom loops.

    **Stateless Prediction**
        fit() learns parameters, predict() applies them without
        modifying state. Can be called repeatedly on different data.

    **Full sklearn Compatibility**
        Single-series design enables full compatibility with sklearn tools:
        pipelines, GridSearchCV, cross_validate, etc.

    Notes
    -----
    For full sklearn compatibility (GridSearchCV, clone, pipelines, etc.),
    estimators should inherit from sklearn.base.BaseEstimator to get
    parameter management methods (get_params, set_params).

    The BaseChangeDetector class combines both requirements automatically.
    """

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the detector to training data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training time series data.
            - Univariate: shape (n_samples, 1)
            - Multivariate: shape (n_samples, n_features)
            Never use 1D arrays, even for univariate data.

        y : None
            Ignored. Exists for sklearn API compatibility (e.g., pipelines).
            Changepoint detection is unsupervised on single series.

        Returns
        -------
        self
            Fitted detector instance (enables method chaining).
        """
        ...

    def predict(self, X: ArrayLike) -> Segmentation:
        """Detect changepoints in a single time series.

        Parameters
        ----------
        X : ArrayLike
            Time series data to analyze for changepoints.
            Must be 2D array of shape (n_samples, n_features).
            - Univariate: shape (n_samples, 1)
            - Multivariate: shape (n_samples, n_features)
            Never use 1D arrays, even for univariate data.

        Returns
        -------
        Segmentation
            Detection result as a dict with required fields:

            - "changepoints": np.ndarray, changepoint locations,
              shape (n_changepoints,)
            - "labels": np.ndarray, segment assignments,
              shape (n_changepoints + 1,)
            - "n_samples": int, number of timepoints analyzed

            Optional fields:

            - "n_features": int, number of features/channels
            - "changed_features": list[np.ndarray], per-changepoint
              feature indices
            - "meta": dict, algorithm-specific metadata

        Examples
        --------
        >>> # Fit and predict on time series
        >>> detector.fit(X_train)
        >>> result = detector.predict(X_test)
        >>> print(result["changepoints"])  # Changepoint locations
        >>> print(result["labels"])  # Segment assignments
        >>> print(len(result["changepoints"]))  # Number of changepoints

        Notes
        -----
        For multiple series, use list comprehension or loops:
        ``results = [detector.predict(X_i) for X_i in series_list]``
        """
        ...
