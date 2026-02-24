"""Base detector implementation providing sklearn compatibility.

Provides a minimal base class for change detectors that:
- Inherits from sklearn.base.BaseEstimator for compatibility
- Defines custom tags via __sklearn_tags__()
- Provides optional transform() convenience method

Subclasses implement fit() and predict() directly without forced private methods.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils._tags import TransformerTags

from skchange.new_api.typing import ArrayLike, Segmentation
from skchange.new_api.utils import ChangeDetectorTags, SkchangeTags, sparse_to_dense


class BaseChangeDetector(BaseEstimator):
    """Base class for change detectors providing sklearn compatibility.

    Inherits from sklearn.base.BaseEstimator for full sklearn compatibility,
    including cloning, GridSearchCV, pipeline support, and get_params/set_params.

    This base class provides:
    - Custom tags via __sklearn_tags__() for change detection metadata
    - Optional transform() method that converts predict() output to dense labels
    - No enforced private methods - implement public API directly

    Subclasses must implement:
    - fit(X, y=None) -> self
    - predict(X) -> Segmentation

    Internal implementation is completely up to the subclass:
    - Use any internal data representation (numpy, pandas, polars, etc.)
    - Apply validation as needed using utils.validate_data() or your own logic
    - Output from predict() must be a Segmentation dict with numpy arrays

    Examples
    --------
    >>> class MyDetector(BaseChangeDetector):
    ...     def __init__(self, threshold=1.0):
    ...         self.threshold = threshold
    ...
    ...     def fit(self, X, y=None):
    ...         # Your validation and fitting logic
    ...         self.n_features_in_ = X.shape[1]
    ...         return self
    ...
    ...     def predict(self, X):
    ...         # Your detection logic
    ...         changepoints = np.array([50, 100])
    ...         return make_segmentation(changepoints=changepoints)
    """

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get estimator tags for change detection.

        Override this method in subclasses to customize tags.

        Returns
        -------
        tags : SkchangeTags
            Estimator tags with change detection specific attributes.

        Examples
        --------
        >>> class MyDetector(BaseChangeDetector):
        ...     def __sklearn_tags__(self):
        ...         tags = super().__sklearn_tags__()
        ...         tags.target_tags.required = True  # Supervised detector
        ...         return tags
        """
        tags = SkchangeTags()
        tags.change_detector_tags = ChangeDetectorTags()
        tags.transformer_tags = TransformerTags(preserves_dtype=[])
        return tags

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform time series to dense segment labels.

        This is an optional convenience method that converts the sparse
        changepoint representation from predict() into dense per-sample labels.

        The default implementation calls predict(X) and converts the result
        to dense labels. Override this method if you want custom behavior.

        Parameters
        ----------
        X : ArrayLike
            Time series data. Expected shape (n_samples, n_features).
            Actual validation depends on your predict() implementation.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Dense segment labels, one per sample.

        Examples
        --------
        >>> detector = MyDetector().fit(X_train)
        >>> labels = detector.transform(X_test)
        >>> # Equivalent to:
        >>> result = detector.predict(X_test)
        >>> labels = sparse_to_dense(result, n_samples=len(X_test))

        Notes
        -----
        This method is not part of the ChangeDetector Protocol - it's purely
        a convenience. You can use predict() and sparse_to_dense() directly
        if you prefer explicit conversion.
        """
        X = np.asarray(X)
        result = self.predict(X)
        return sparse_to_dense(result, n_samples=len(X))

    def fit_transform(self, X, y: ArrayLike | None = None, **fit_params) -> np.ndarray:
        """Fit to data, then transform it.

        This is a convenience method for sklearn compatibility.
        Equivalent to calling fit(X, y).transform(X).

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series data.

        y : None
            Ignored. Exists for sklearn API compatibility.

        **fit_params : dict
            Additional parameters passed to fit(). Not commonly used,
            but provided for sklearn compatibility.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Dense segment labels.

        Examples
        --------
        >>> detector = MyDetector()
        >>> labels = detector.fit_transform(X_train)
        >>> # Equivalent to:
        >>> detector.fit(X_train)
        >>> labels = detector.transform(X_train)

        Notes
        -----
        The **fit_params argument exists for sklearn compatibility (e.g.,
        passing sample_weight through pipelines), but most change detectors
        won't use it.
        """
        return self.fit(X, y, **fit_params).transform(X)

    def fit_predict(self, X, y: ArrayLike | None = None, **fit_params) -> Segmentation:
        """Fit to data, then predict changepoints.

        This is a convenience method for sklearn compatibility.
        Equivalent to calling fit(X, y).predict(X).

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series data.

        y : None
            Ignored. Exists for sklearn API compatibility.

        **fit_params : dict
            Additional parameters passed to fit(). Not commonly used,
            but provided for sklearn compatibility.

        Returns
        -------
        result : dict
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
        >>> detector = MyDetector()
        >>> result = detector.fit_predict(X_train)
        >>> print(result["changepoints"])
        >>> # Equivalent to:
        >>> detector.fit(X_train)
        >>> result = detector.predict(X_train)

        Notes
        -----
        This is similar to fit_transform() but returns sparse Segmentation
        dict instead of dense labels. Use fit_transform() if you need
        dense labels for sklearn compatibility.
        """
        return self.fit(X, y, **fit_params).predict(X)


# TODO: Finish tags.
# TODO: Add support for parameter specification. To simplify hyperparameter tuning
#       and validation.
