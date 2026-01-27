"""Reference base detector implementation for single/multiple series handling.

This module demonstrates the recommended pattern for handling both single and
multiple time series in a clean, maintainable way.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
from sklearn.base import BaseEstimator

from skchange.new_api.typing import ArrayLike, Segmentation
from skchange.new_api.utils import SkchangeTags, sparse_to_dense, validate_data


class BaseChangeDetector(BaseEstimator):
    """Base class for change detectors with automatic single/multi series handling.

    Inherits from sklearn.base.BaseEstimator for full sklearn compatibility,
    including cloning, GridSearchCV, and pipeline support.

    This base class provides dispatching logic for fit() to handle both single
    and multiple series. The predict() and transform() methods operate on single
    series only - use list comprehensions for batch prediction.

    The transform() method is provided as a convenience and has a default
    implementation that converts sparse changepoint results to dense labels.
    It is not required by the ChangeDetector Protocol.

    Implementation Guide for Subclasses
    ------------------------------------

    1. **Single-series only detector** (e.g., PELT):
       - Override `__sklearn_tags__()` and set
         `tags.change_detector_tags.capability_multiple_series = False`
       - Implement `_fit(X: ArrayLike, y)` and `_predict(X: ArrayLike)`
       - Users get clear error if they try to pass multiple series to fit()

    2. **Multiple-series detector** (e.g., batch processor):
       - Override `__sklearn_tags__()` and set
         `tags.change_detector_tags.capability_multiple_series = True`
       - Implement `_fit_multiple(X: list[ArrayLike], y)` for batch learning
       - Implement `_predict(X: ArrayLike)` for per-series prediction
       - Optionally implement `_fit(X, y)` to handle single series efficiently

    3. **Universal detector** (works on both):
       - Override `__sklearn_tags__()` and set
         `tags.change_detector_tags.capability_multiple_series = True`
       - Implement `_fit(X, y)` for single series (core logic)
       - Don't override `_fit_multiple` - base class will call `_fit` on each
       - Or implement both for optimized batch processing

    Tags
    ----
    change_detector_tags.capability_multiple_series : bool, default=False
        Whether the detector can process multiple series in a single fit() call.

    Examples
    --------
    >>> # Customizing tags in a subclass
    >>> class MyDetector(BaseChangeDetector):
    ...     def __sklearn_tags__(self):
    ...         tags = super().__sklearn_tags__()
    ...         tags.change_detector_tags.capability_multiple_series = True
    ...         return tags
    """

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get estimator tags.

        Returns
        -------
        tags : Tags
            Estimator tags with change detection specific attributes.
        """
        tags_orig = super().__sklearn_tags__()
        tags_orig.estimator_type = None
        as_dict = {
            field.name: getattr(tags_orig, field.name) for field in fields(tags_orig)
        }
        return SkchangeTags(**as_dict)

    # ==================== Public API ====================

    def fit(
        self,
        X: ArrayLike | list[ArrayLike],
        y: Segmentation | list[Segmentation] | ArrayLike | None = None,
    ) -> BaseChangeDetector:
        """Fit the detector on single or multiple series.

        Parameters
        ----------
        X : ArrayLike | list[ArrayLike]
            Training data.
            - If ArrayLike: shape (n_samples, n_features) for single series
            - If list: each element has shape (n_samples_i, n_features)
        y : Segmentation | list[Segmentation] | ArrayLike | None
            Supervised labels (optional). Formats:

            **Sparse-first**: Only Segmentation dicts accepted for segment
            labels.

            **Segmentation dict** has required fields:

            - "changepoints": np.ndarray of changepoint indices
            - "labels": np.ndarray of segment labels
            - "n_samples": int, number of samples

            Accepted formats:

            - **Segmentation**: Sparse segment labels for single series
            - **list[Segmentation]**: Sparse labels per series
            - **ArrayLike**: Series-level labels ONLY (one label per series,
              for classification tasks). Shape (n_series,).
            - **None**: Unsupervised (most common)

            See ChangeDetector Protocol for detailed format specification.

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
            tags = self.__sklearn_tags__()
            if not tags.change_detector_tags.capability_multiple_series:
                raise ValueError(
                    f"{self.__class__.__name__} does not support "
                    f"multiple series. Please fit on each series "
                    f"separately or use a detector with "
                    f"'change_detector_tags.capability_multiple_series' "
                    f"tag set to True."
                )

            # Validate all series are 2D
            X = [validate_data(X_i) for X_i in X]

            # Validate and normalize y structure
            if y is not None:
                if isinstance(y, list):
                    # Per-series segment labels (must be Segmentation dicts)
                    if len(y) != len(X):
                        raise ValueError(
                            f"If y is a list, it must have same length as X. "
                            f"Got len(X)={len(X)}, len(y)={len(y)}"
                        )
                    # Validate each y_i is Segmentation and convert to dense
                    y_validated = []
                    for i, (y_i, X_i) in enumerate(zip(y, X)):
                        if y_i is not None:
                            if not isinstance(y_i, dict):
                                raise TypeError(
                                    f"y[{i}] must be a Segmentation dict. "
                                    f"Got {type(y_i).__name__}. "
                                    f"Use dense_to_sparse() to convert dense "
                                    f"labels or pass ArrayLike for series-level "
                                    f"classification (not per-series labels)."
                                )
                            # Convert Segmentation to dense labels
                            y_i = sparse_to_dense(y_i)
                            if len(y_i) != len(X_i):
                                raise ValueError(
                                    f"y[{i}] must have same length as "
                                    f"X[{i}]. Got len(y[{i}])={len(y_i)}, "
                                    f"len(X[{i}])={len(X_i)}"
                                )
                        y_validated.append(y_i)
                    y = y_validated
                else:
                    # Series-level classification - one label per series
                    y = np.asarray(y)
                    if y.ndim != 1:
                        raise ValueError(
                            f"If X is a list and y is not a list, y should be "
                            f"1D (one label per series for classification). "
                            f"Got y.shape={y.shape}"
                        )
                    if len(y) != len(X):
                        raise ValueError(
                            f"y must have one label per series. "
                            f"Got len(X)={len(X)}, len(y)={len(y)}"
                        )

            return self._fit_multiple(X, y)
        else:
            # Single series
            X = validate_data(X)
            # Validate y if provided - must be Segmentation
            if y is not None:
                if not isinstance(y, dict):
                    raise TypeError(
                        f"For single series, y must be a Segmentation dict. "
                        f"Got {type(y).__name__}. "
                        f"Use dense_to_sparse() to convert dense labels, or "
                        f"use make_segmentation(changepoints=[...], n_samples=...)."
                    )
                # Convert Segmentation to dense labels
                y = sparse_to_dense(y)
                if len(y) != len(X):
                    raise ValueError(
                        f"y must have same length as X. "
                        f"Got len(y)={len(y)}, len(X)={len(X)}"
                    )
            return self._fit(X, y)

    def predict(self, X: ArrayLike) -> Segmentation:
        """Detect changepoints in a single time series.

        Parameters
        ----------
        X : ArrayLike
            Time series data to analyze. Must be 2D: (n_samples, n_features).

        Returns
        -------
        Segmentation
            Detection result as a dict with required fields:

            - "changepoints": np.ndarray, changepoint indices
            - "labels": np.ndarray, segment labels
            - "n_samples": int, number of samples

            Optional fields: "n_features", "scores", "affected_variables",
            "meta".

        Examples
        --------
        >>> result = detector.predict(X)
        >>> print(result["changepoints"])
        >>> # For multiple series, use list comprehension
        >>> results = [detector.predict(Xi) for Xi in series_list]
        """
        X = validate_data(X)
        return self._predict(X)

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform to dense segment labels.

        This is an optional convenience method that converts the sparse
        changepoint representation from predict() into dense segment labels.

        Parameters
        ----------
        X : ArrayLike
            Data to transform. Must be 2D: (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Segment labels, shape (n_samples,).

        Examples
        --------
        >>> labels = detector.transform(X)
        >>> # For multiple series, use list comprehension
        >>> labels_list = [detector.transform(Xi) for Xi in series_list]
        """
        X = validate_data(X)
        return self._transform(X)

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

    def _predict(self, X: ArrayLike) -> Segmentation:
        """Predict on a single series. Override in subclasses.

        Parameters
        ----------
        X : ArrayLike
            Shape (n_timepoints, n_channels). Already validated as 2D.

        Returns
        -------
        Segmentation
            Detection result as a dict with required fields:

            - "changepoints": np.ndarray, changepoint indices
            - "labels": np.ndarray, segment labels
            - "n_samples": int, number of samples

            Optional fields: "n_features", "scores", "affected_variables",
            "meta".
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _predict() method"
        )

    def _transform(self, X: ArrayLike) -> np.ndarray:
        """Transform single series to dense labels. Override if needed.

        Parameters
        ----------
        X : ArrayLike
            Shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Dense segment labels, shape (n_samples,).
        """
        # Default: convert sparse predictions to dense
        prediction = self._predict(X)
        return sparse_to_dense(prediction)

    def _fit_multiple(
        self,
        X: list[ArrayLike],
        y: list[ArrayLike] | ArrayLike | None = None,
    ) -> BaseChangeDetector:
        """Fit on multiple series. Override for batch-specific logic.

        Default behavior: fit independently on each series (stateless detector).

        Parameters
        ----------
        X : list[ArrayLike]
            List of series, each (n_samples_i, n_features).
        y : list[ArrayLike] | ArrayLike | None
            Optional labels - already validated and converted to dense in fit().
            Segmentation dicts are automatically converted to dense arrays.
            Either list (per-timepoint for each series) or array (one label
            per series).

        Returns
        -------
        self
        """
        # Default: fit on each series independently
        # Override this for detectors that learn shared parameters
        for i, X_i in enumerate(X):
            if y is not None:
                y_i = y[i] if isinstance(y, list) else y[i]
            else:
                y_i = None
            self._fit(X_i, y_i)
        return self
