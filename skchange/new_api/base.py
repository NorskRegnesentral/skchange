"""Reference base detector implementation for single/multiple series handling.

This module demonstrates the recommended pattern for handling both single and
multiple time series in a clean, maintainable way.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from .typing import ArrayLike, ChangeDetectionResult


class BaseChangeDetector(BaseEstimator):
    """Base class for change detectors with automatic single/multi series handling.

    Inherits from sklearn.base.BaseEstimator for full sklearn compatibility,
    including cloning, GridSearchCV, and pipeline support.

    This base class provides the dispatching logic so that concrete detectors
    only need to implement the methods relevant to their capabilities.

    The transform() method is provided as a convenience and has a default
    implementation that converts sparse changepoint results to dense labels.
    It is not required by the ChangeDetector Protocol.

    Implementation Guide for Subclasses
    ------------------------------------

    1. **Single-series only detector** (e.g., PELT):
       - Set `_tags = {"capability:multiple_series": False}`
       - Implement `_fit(X: ArrayLike, y)` and `_predict(X: ArrayLike)`
       - Users get clear error if they try to pass multiple series

    2. **Multiple-series detector** (e.g., batch processor):
       - Set `_tags = {"capability:multiple_series": True}`
       - Implement `_fit_multiple(X: list[ArrayLike], y)` and
         `_predict_multiple(X: list[ArrayLike])`
       - Optionally implement `_fit(X, y)` to handle single series efficiently
       - Base class can wrap single series in list if you don't implement `_fit`

    3. **Universal detector** (works on both):
       - Set `_tags = {"capability:multiple_series": True}`
       - Implement `_fit(X, y)` for single series (core logic)
       - Don't override `_fit_multiple` - base class will call `_fit` on each
       - Or implement both for optimized batch processing

    Tags
    ----
    capability:multiple_series : bool, default=False
        Whether the detector can process multiple series in a single fit.
    """

    _tags = {
        "capability:multiple_series": False,
    }

    def get_tag(self, tag_name: str, raise_error: bool = False) -> Any:
        """Get the value of a tag."""
        if raise_error and tag_name not in self._tags:
            raise ValueError(f"Tag {tag_name} not found")
        return self._tags.get(tag_name)

    # ==================== Public API ====================

    def fit(
        self,
        X: ArrayLike | list[ArrayLike],
        y: ArrayLike | list[ArrayLike] | None = None,
    ) -> BaseChangeDetector:
        """Fit the detector on single or multiple series.

        Parameters
        ----------
        X : ArrayLike | list[ArrayLike]
            Training data.
            - If ArrayLike: shape (n_samples, n_features) for single series
            - If list: each element has shape (n_samples_i, n_features)
        y : ArrayLike | list[ArrayLike] | None, optional
            Supervised labels (rarely used). Interpretation depends on types:

            - If X is ArrayLike and y is ArrayLike: per-timepoint labels
            - If X is list and y is ArrayLike: one label per series
            - If X is list and y is list: per-timepoint labels for each series
            - If y is None: unsupervised (most common)

        Returns
        -------
        self
            Fitted detector instance.

        Raises
        ------
        ValueError
            If detector doesn't support multiple series but receives a list.
        """
        if isinstance(X, list):
            # Multiple series requested
            if not self.get_tag("capability:multiple_series"):
                raise ValueError(
                    f"{self.__class__.__name__} does not support multiple series. "
                    f"Please fit on each series separately or use a detector with "
                    f"'capability:multiple_series' tag set to True."
                )

            # Validate all series are 2D
            X = [self._validate_2d(X_i) for X_i in X]

            # Validate y structure
            if y is not None:
                if isinstance(y, list):
                    # Per-timepoint labels for each series
                    if len(y) != len(X):
                        raise ValueError(
                            f"If y is a list, it must have same length as X. "
                            f"Got len(X)={len(X)}, len(y)={len(y)}"
                        )
                    # Could validate lengths match X[i] here if needed
                else:
                    # One label per series - validate shape
                    y = np.asarray(y)
                    if y.ndim != 1:
                        raise ValueError(
                            f"If X is a list and y is not a list, y should be 1D "
                            f"(one label per series). Got y.shape={y.shape}"
                        )
                    if len(y) != len(X):
                        raise ValueError(
                            f"y must have one label per series. "
                            f"Got len(X)={len(X)}, len(y)={len(y)}"
                        )

            return self._fit_multiple(X, y)
        else:
            # Single series
            X = self._validate_2d(X)
            return self._fit(X, y)

    def predict(self, X: ArrayLike | list[ArrayLike]) -> list[ChangeDetectionResult]:
        """Detect changepoints in single or multiple series.

        Always returns list of ChangeDetectionResult.

        Parameters
        ----------
        X : ArrayLike | list[ArrayLike]
            Data to analyze.

        Returns
        -------
        list[ChangeDetectionResult]
            List of detection results.
            - Single series: List with 1 element
            - Multiple series: List with N elements
        """
        if isinstance(X, list):
            if not self.get_tag("capability:multiple_series"):
                raise ValueError(
                    f"{self.__class__.__name__} does not support multiple series"
                )
            X = [self._validate_2d(X_i) for X_i in X]
            return self._predict_multiple(X)
        else:
            # Single series - wrap in list for consistent output
            X = self._validate_2d(X)
            single_result = self._predict(X)
            return [single_result]

    def transform(self, X: ArrayLike | list[ArrayLike]) -> list[ArrayLike]:
        """Transform to dense segment labels.

        This is an optional convenience method that converts the sparse
        changepoint representation from predict() into dense segment labels.

        Always returns a list, regardless of input type.

        Parameters
        ----------
        X : ArrayLike | list[ArrayLike]
            Data to transform.

        Returns
        -------
        list[ArrayLike]
            List of segment labels.
            - Single series: List with 1 element
            - Multiple series: List with N elements
        """
        if isinstance(X, list):
            if not self.get_tag("capability:multiple_series"):
                raise ValueError(
                    f"{self.__class__.__name__} does not support multiple series"
                )
            X = [self._validate_2d(X_i) for X_i in X]
            return self._transform_multiple(X)
        else:
            # Single series - wrap in list for consistent output
            X = self._validate_2d(X)
            return [self._transform(X)]

    # ==================== Methods to Override ====================

    def _fit(self, X: ArrayLike, y: ArrayLike | None = None) -> BaseChangeDetector:
        """Fit on a single series. Override in subclasses.

        Parameters
        ----------
        X : ArrayLike
            Shape (n_samples, n_features). Already validated as 2D.
        y : ArrayLike | None
            Optional labels.

        Returns
        -------
        self
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _fit() method"
        )

    def _predict(self, X: ArrayLike) -> ChangeDetectionResult:
        """Predict on a single series. Override in subclasses.

        Parameters
        ----------
        X : ArrayLike
            Shape (n_timepoints, n_channels). Already validated as 2D.

        Returns
        -------
        ChangeDetectionResult
            Detection results for single series.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _predict() method"
        )

    def _transform(self, X: ArrayLike) -> ArrayLike:
        """Transform single series to dense labels. Override if needed.

        Parameters
        ----------
        X : ArrayLike
            Shape (n_samples, n_features).

        Returns
        -------
        ArrayLike
            Dense segment labels, shape (n_samples,).
        """
        # Default: convert sparse predictions to dense
        prediction = self._predict(X)
        return self._sparse_to_dense(prediction)

    def _fit_multiple(
        self,
        X: list[ArrayLike],
        y: list[ArrayLike] | None = None,
    ) -> BaseChangeDetector:
        """Fit on multiple series. Override for batch-specific logic.

        Default behavior: fit independently on each series (stateless detector).

        Parameters
        ----------
        X : list[ArrayLike]
            List of series, each (n_samples_i, n_features).
        y : list[ArrayLike] | None
            Optional labels for each series.

        Returns
        -------
        self
        """
        # Default: fit on each series independently
        # Override this for detectors that learn shared parameters
        for i, X_i in enumerate(X):
            y_i = y[i] if y is not None else None
            self._fit(X_i, y_i)
        return self

    def _predict_multiple(self, X: list[ArrayLike]) -> list[ChangeDetectionResult]:
        """Predict on multiple series. Override for batch-specific logic.

        Default behavior: predict independently on each series.

        Parameters
        ----------
        X : list[ArrayLike]
            List of series to analyze.

        Returns
        -------
        list[ChangeDetectionResult]
            List of detection results for all series.
        """
        # Default: predict on each series independently
        return [self._predict(X_i) for X_i in X]

    def _transform_multiple(self, X: list[ArrayLike]) -> list[ArrayLike]:
        """Transform multiple series. Override for batch-specific logic.

        Default behavior: transform independently.

        Parameters
        ----------
        X : list[ArrayLike]
            List of series.

        Returns
        -------
        list[ArrayLike]
            Dense labels for each series.
        """
        return [self._transform(X_i) for X_i in X]

    # ==================== Utility Methods ====================

    def _validate_2d(self, X: ArrayLike) -> ArrayLike:
        """Ensure X is 2D array (n_samples, n_features).

        Parameters
        ----------
        X : ArrayLike
            Input data.

        Returns
        -------
        ArrayLike
            2D array.

        Raises
        ------
        ValueError
            If X cannot be converted to 2D or has wrong number of dimensions.
        """
        X = np.asarray(X)

        if X.ndim == 1:
            # Univariate series given as 1D - reshape to (n, 1)
            X = X.reshape(-1, 1)
        elif X.ndim == 2:
            # Already 2D - good
            pass
        else:
            raise ValueError(
                f"Expected 1D or 2D array, got {X.ndim}D array with shape {X.shape}. "
                f"For single series use shape (n_timepoints, n_channels). "
                f"For multiple series pass a list of 2D arrays."
            )

        return X

    def _sparse_to_dense(self, result: ChangeDetectionResult) -> ArrayLike:
        """Convert sparse prediction to dense segment labels.

        Parameters
        ----------
        result : ChangeDetectionResult
            Sparse changepoint locations with n_samples.

        Returns
        -------
        ArrayLike
            Dense labels, shape (n_samples,).
        """
        # Simple implementation - override in subclasses for specific formats
        n_samples = result["n_samples"]
        labels = np.zeros(n_samples, dtype=int)

        if result["indices"] is not None:
            changepoints = np.asarray(result["indices"])
            if len(changepoints) > 0:
                # Create segment labels
                changepoints = np.concatenate([[0], changepoints, [n_samples]])
                for seg_id in range(len(changepoints) - 1):
                    start = changepoints[seg_id]
                    end = changepoints[seg_id + 1]
                    labels[start:end] = seg_id

        return labels
