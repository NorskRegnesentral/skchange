"""Typing definitions for the new skchange API.

Design Philosophy
-----------------
Single vs. Multiple Series Handling:

1. **Input Convention:**
   - Single series: ArrayLike of shape (n_samples, n_features)
   - Multiple series: list[ArrayLike], each with shape (n_samples_i, n_features)
   - Univariate series use n_features=1 (never 1D arrays)

2. **Protocol Design:**
   - Public API accepts both: ArrayLike | list[ArrayLike]
   - Concrete detectors implement only what they support
   - Base class handles dispatching automatically

3. **Implementation Pattern:**
   - Detectors override `_fit(X: ArrayLike)` for single series
   - Detectors override `_fit_multiple(X: list[ArrayLike])` for batch processing
   - Universal detectors can implement both or rely on default composition
   - Use tags to declare capabilities explicitly

This avoids API fragmentation while keeping implementations simple.
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


class ChangeDetectionResult(TypedDict):
    """Result of changepoint detection on a single time series.

    This is the fundamental output type. Both single and multiple series
    predictions return list[ChangeDetectionResult].

    TypedDict allows plain dicts to be returned, avoiding custom classes and
    maintaining alignment with scikit-learn's design philosophy.

    The result contains a sparse representation with per-changepoint arrays
    (indices, scores, affected_variables), per-segment arrays (segment_labels),
    and scalar metadata (n_samples, n_features).

    Required Fields
    ---------------
    indices : np.ndarray of shape (n_changepoints,)
        Changepoint indices where the time series exhibits structural breaks.
        Integer array with values in [0, n_samples). Each index marks the start
        of a new segment. Empty array if no changepoints detected.
        dtype: typically int64, but any integer dtype accepted.

    segment_labels : np.ndarray of shape (n_changepoints + 1,)
        Segment labels for each interval between changepoints (and boundaries).
        Integer array starting from 0. Default is [0, 1, 2, ...] where each
        segment has a unique label. Can assign the same label to multiple
        segments to group them into the same regime/state.
        dtype: typically int64, but any integer dtype accepted.

    n_samples : int
        Number of samples (timepoints) in the analyzed time series.
        Required for sparse-to-dense conversion and validation.
        Must be positive integer.

    n_features : int
        Number of features (variables/channels) in the analyzed time series.
        Must be positive integer, use 1 for univariate data.

    Optional Fields
    ---------------
    scores : np.ndarray of shape (n_changepoints,)
        Score, strength, or confidence measure for each detected changepoint.
        Higher values typically indicate stronger evidence for the changepoint.
        Scale and interpretation are detector-specific.
        dtype: typically float64, but any numeric dtype accepted.

    affected_variables : list[np.ndarray]
        Variable/channel indices affected at each changepoint.
        List of length n_changepoints. Each element is an integer array of
        shape (n_affected_i,) containing indices in [0, n_features).
        None or omit entirely if not applicable (e.g., univariate data) or
        if all variables are affected at all changepoints.
        dtype per array: typically int64, but any integer dtype accepted.

    meta : dict[str, Any]
        Optional metadata about the detection process or results.
        May include algorithm parameters, thresholds, convergence info,
        computation time, or any detector-specific information.
        Structure is detector-dependent and not standardized.

    Examples
    --------
    >>> # Basic univariate detection (use helper function)
    >>> result = make_change_detection_result(
    ...     indices=np.array([10, 50, 90]),
    ...     segment_labels=np.array([0, 1, 2, 3]),
    ...     n_samples=200,
    ...     n_features=1,
    ...     scores=np.array([0.9, 0.8, 0.7]),
    ... )

    >>> # Or create dict directly
    >>> result = {
    ...     "indices": np.array([10, 50, 90]),
    ...     "segment_labels": np.array([0, 1, 2, 3]),
    ...     "n_samples": 200,
    ...     "n_features": 1,
    ...     "scores": np.array([0.9, 0.8, 0.7]),
    ... }

    >>> # Access fields via dict keys
    >>> changepoints = result["indices"]
    >>> labels = result["segment_labels"]
    """

    indices: np.ndarray
    segment_labels: np.ndarray
    n_samples: int
    n_features: int
    scores: NotRequired[np.ndarray]
    affected_variables: NotRequired[list[np.ndarray]]
    meta: NotRequired[dict[str, Any]]


@runtime_checkable
class ChangeDetector(Protocol):
    """Protocol for changepoint detection algorithms.

    Defines the minimal public interface for changepoint detectors supporting
    both single and multiple time series. Only fit() and predict() are required.

    Terminology
    -----------
    **Univariate vs Multivariate** (feature dimension):

    - Univariate: n_features = 1 (single channel/variable)
    - Multivariate: n_features > 1 (multiple channels/variables)

    **Single vs Multiple Series** (for fit only):

    - Single: One time series, shape (n_samples, n_features)
    - Multiple: List of time series, each (n_samples_i, n_features).
      Series may have different lengths (n_samples_i) but must share
      the same number of features (n_features) across all series.

    Design Principles
    -----------------
    **Intentional Asymmetry**
        fit() accepts single or multiple series (enabling shared parameter
        learning across series), while predict() accepts only single
        series (per-series operation without cross-series computation).
        This design reflects the semantic difference between training
        and inference.

    **Simple Output**
        predict() returns one ChangeDetectionResult dict per input
        series. Use e.g. list comprehension for multiple series prediction.

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

    def fit(
        self,
        X: ArrayLike | list[ArrayLike],
        y: ArrayLike | dict | list[ArrayLike | dict] | None = None,
    ) -> Self:
        """Fit the detector to training data.

        Parameters
        ----------
        X : ArrayLike | list[ArrayLike]
            Training data.

            - Single series: 2D array of shape (n_samples, n_features)
            - Multiple series: List of 2D arrays, each
              (n_samples_i, n_features)
            - Univariate data: Use (n_samples, 1), never 1D arrays

        y : ArrayLike | dict | list[ArrayLike | dict] | None, default=None
            Supervised labels (optional). Default is None (unsupervised).

            Single series (X is ArrayLike):

            - ArrayLike: Dense segment labels, shape (n_samples,).
              Example: ``np.array([0,0,0,1,1,1,2,2])``
            - dict: Sparse changepoint format with keys:

              - ``"indices"``: integer changepoint locations, shape (n_changepoints,)
              - ``"segment_labels"``: integer segment labels,
                shape (n_changepoints + 1,)

              Example: ``{"indices": np.array([50, 100]),
              "segment_labels": np.array([0, 1, 2])}``

            Multiple series (X is list[ArrayLike]):

            - ArrayLike: One label per series, shape (n_series,)
            - list: Per-series labels, each element ArrayLike or dict
              as above

        Returns
        -------
        self
            Fitted detector instance (enables method chaining).
        """
        ...

    def predict(self, X: ArrayLike) -> ChangeDetectionResult:
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
        ChangeDetectionResult
            Detection result as a dict containing:
            - "indices": Changepoint locations, shape (n_changepoints,)
            - "segment_labels": Segment assignments, shape (n_changepoints + 1,)
            - "n_samples": Number of timepoints analyzed
            - "n_features": Number of features/channels
            - "scores" (optional): Changepoint scores, shape (n_changepoints,)
            - "affected_variables" (optional): Per-changepoint variable indices
            - "meta" (optional): Algorithm-specific metadata

        Examples
        --------
        >>> # Predict on single series
        >>> detector.fit(X_train)
        >>> result = detector.predict(X_test)
        >>> print(result["indices"])  # Changepoint locations
        >>> print(result["segment_labels"])  # Segment assignments
        >>> print(len(result["indices"]))  # Number of changepoints

        >>> # Predict on multiple series using list comprehension
        >>> results = [detector.predict(X) for X in test_series]
        >>> for i, result in enumerate(results):
        ...     print(f"Series {i}: {len(result['indices'])} changepoints")
        """
        ...
