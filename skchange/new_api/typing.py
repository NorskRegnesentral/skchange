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

from typing import Protocol, TypedDict, runtime_checkable

import numpy as np

# Type alias for array-like inputs
# We use Any instead of numpy.typing.ArrayLike to avoid linter warnings about
# potentially compatible array types (e.g., pandas DataFrames, polars, xarray)
# that aren't covered by numpy's definition.
# Convention: ArrayLike is for INPUTS (flexible), np.ndarray for OUTPUTS (consistent).
ArrayLike = np.typing.ArrayLike


class Segmentation(TypedDict):
    """Sparse segmentation representation for changepoint detection.

    This is the universal sparse format used throughout the API for both
    input (y labels) and output (predict results).

    TypedDict allows plain dicts to be returned, avoiding custom classes and
    maintaining alignment with scikit-learn's design philosophy.

    The result contains a sparse representation focused on what was detected:
    changepoint locations, optional segment labels, and optional per-changepoint
    feature information.

    Required Fields
    ---------------
    changepoints : np.ndarray of shape (n_changepoints,)
        Changepoint indices where the time series exhibits structural breaks.
        Integer array with values in [0, n_samples). Each index marks the start
        of a new segment. Empty array if no changepoints detected.
        dtype: typically int64, but any integer dtype accepted.

    Optional Fields
    ---------------
    labels : np.ndarray of shape (n_changepoints + 1,)
        Segment labels for each interval between changepoints (and boundaries).
        Integer array starting from 0.
        **If omitted, automatically generated as [0, 1, 2, ...] where each
        segment has a unique label.** Explicitly provide labels to group
        segments into the same regime/state (e.g., for recurring patterns).
        dtype: typically int64, but any integer dtype accepted.

    changed_features : list[np.ndarray]
        Feature/channel indices affected at each changepoint.
        List of length n_changepoints. Each element is an integer array of
        shape (n_affected_i,) containing indices in [0, n_features).
        None or omit entirely if not applicable (e.g., univariate data) or
        if all features are affected at all changepoints.
        dtype per array: typically int64, but any integer dtype accepted.

    Examples
    --------
    >>> # Minimal result - just changepoints
    >>> result = make_segmentation(
    ...     changepoints=np.array([10, 50, 90]),
    ... )
    >>> "labels" in result  # Not included unless provided
    False
    >>> result["changepoints"]
    array([10, 50, 90])

    >>> # Explicit labels for recurring patterns
    >>> result = make_segmentation(
    ...     changepoints=np.array([10, 50, 90]),
    ...     labels=np.array([0, 1, 0, 1]),  # Alternating states
    ... )
    >>> result["labels"]
    array([0, 1, 0, 1])

    >>> # Full result with all optional fields
    >>> result = make_segmentation(
    ...     changepoints=np.array([10, 50]),
    ...     labels=np.array([0, 1, 2]),
    ...     changed_features=[np.array([0, 1]), np.array([2])],
    ... )

    >>> # Labels generated lazily when converting to dense
    >>> from skchange.new_api.utils import sparse_to_dense
    >>> result = make_segmentation(changepoints=np.array([50]))
    >>> dense_labels = sparse_to_dense(result, n_samples=100)  # Auto-generates [0, 1]
    """

    changepoints: np.ndarray
    labels: NotRequired[np.ndarray]
    changed_features: NotRequired[list[np.ndarray]]


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

        y : ArrayLike | None, default=None
            Exists for sklearn API compatibility (e.g. pipelines).
            Most changepoint detectors are unsupervised on single series,
            so y is typically None and ignored.

        Returns
        -------
        self
            Fitted detector instance.
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
        dict
            Detection result as a dict with required fields:

            - "changepoints": np.ndarray, changepoint locations,
              shape (n_changepoints,)

            Optional fields:

            - "labels": np.ndarray, segment assignments,
              shape (n_changepoints + 1,)
            - "changed_features": list[np.ndarray], per-changepoint
              feature indices

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
