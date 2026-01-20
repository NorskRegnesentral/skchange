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

    # Required fields - type checkers enforce these
    indices: np.ndarray
    segment_labels: np.ndarray
    n_samples: int
    n_features: int
    # Optional fields - can be omitted
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

    **Single vs Multiple** (series dimension):
        - Single: One time series, shape (n_samples, n_features)
        - Multiple: List of time series, each (n_samples_i, n_features)
          Series may have different lengths (n_samples_i) but must share
          the same number of features (n_features) across all series.

    These dimensions are independent - all four combinations are valid:
        - Single univariate: array of shape (n_samples, 1)
        - Single multivariate: array of shape (n_samples, n_features)
        - Multiple univariate: list of arrays, each (n_samples_i, 1)
        - Multiple multivariate: list of arrays, each (n_samples_i, n_features)

    Design Principles
    -----------------
    **Unified API**: Same interface handles single or multiple series via type unions.

    **Consistent Output**: predict() always returns list[ChangeDetectionResult],
    regardless of input type (single series → list of 1 element, multiple series →
    list of N elements). This consistency simplifies downstream code.

    **Stateless Prediction**: fit() learns parameters, predict() is stateless and
    can be called repeatedly on different data.
    """

    def fit(
        self,
        X: ArrayLike | list[ArrayLike],
        y: ArrayLike | list[ArrayLike] | None = None,
    ) -> Self:
        """Fit the detector to training data.

        Parameters
        ----------
        X : ArrayLike | list[ArrayLike]
            Training data.
            - Single series: 2D array of shape (n_samples, n_features)
            - Multiple series: List of 2D arrays, each (n_samples_i, n_features)
            - Univariate data: Use (n_samples, 1), never 1D arrays

        y : ArrayLike | list[ArrayLike] | None, default=None
            Supervised labels (optional).
            Structure depends on X type:

            If X is ArrayLike (single series):
                - y is ArrayLike: Per-timepoint labels, shape (n_samples,)
                - y is None: Unsupervised (most common)

            If X is list[ArrayLike] (multiple series):
                - y is ArrayLike: One label per series, shape (n_series,)
                  Example: Binary classification of each series
                - y is list[ArrayLike]: Per-timepoint labels for each series
                  Example: y[i] has shape (len(X[i]),) matching X[i]
                - y is None: Unsupervised (most common)

        Returns
        -------
        self
            Fitted detector instance (enables method chaining).
        """
        ...

    def predict(self, X: ArrayLike | list[ArrayLike]) -> list[ChangeDetectionResult]:
        """Detect changepoints in time series data.

        Parameters
        ----------
        X : ArrayLike | list[ArrayLike]
            Data to analyze for changepoints.
            - Single series: 2D array of shape (n_samples, n_features)
            - Multiple series: List of 2D arrays, each (n_samples_i, n_features)
            - Univariate data: Use (n_samples, 1), never 1D arrays

        Returns
        -------
        list[ChangeDetectionResult]
            Detection results, one per input series. Always returns a list,
            regardless of input type.
            - Single series input: List with 1 element
            - Multiple series input: List with N elements

            Each result is a dict (ChangeDetectionResult) containing:
            - "indices": Changepoint locations, shape (n_changepoints,)
            - "segment_labels": Segment assignments, shape (n_changepoints + 1,)
            - "n_samples": Number of timepoints analyzed
            - "n_features": Number of features/channels
            - "scores" (optional): Changepoint scores, shape (n_changepoints,)
            - "affected_variables" (optional): Per-changepoint variable indices
            - "meta" (optional): Algorithm-specific metadata

        Examples
        --------
        >>> # Single series prediction
        >>> detector.fit(X_train)
        >>> results = detector.predict(X_test)
        >>> print(results[0]["indices"])  # Changepoint locations
        >>> print(results[0]["segment_labels"])  # Segment assignments

        >>> # Multiple series prediction
        >>> results = detector.predict([X1, X2, X3])
        >>> for i, result in enumerate(results):
        ...     print(f"Series {i}: {len(result['indices'])} changepoints")
        """
        ...
