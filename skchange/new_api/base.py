"""Base detector implementation providing sklearn compatibility.

Provides a minimal base class for change detectors that:
- Inherits from sklearn.base.BaseEstimator for compatibility
- Defines custom tags via __sklearn_tags__()
- Provides optional transform() convenience method

Subclasses implement fit() and predict() directly without forced private methods.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
from sklearn.base import BaseEstimator

from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils import SkchangeTags, sparse_to_dense


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
    ...         return make_segmentation(
    ...             changepoints=changepoints,
    ...             n_samples=len(X),
    ...             n_features=X.shape[1]
    ...         )
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
        ...         tags.change_detector_tags.capability_multiple_series = True
        ...         return tags
        """
        tags_orig = super().__sklearn_tags__()
        tags_orig.estimator_type = None
        as_dict = {
            field.name: getattr(tags_orig, field.name) for field in fields(tags_orig)
        }
        return SkchangeTags(**as_dict)

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
        >>> labels = sparse_to_dense(result)

        Notes
        -----
        This method is not part of the ChangeDetector Protocol - it's purely
        a convenience. You can use predict() and sparse_to_dense() directly
        if you prefer explicit conversion.
        """
        result = self.predict(X)
        return sparse_to_dense(result)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        This method is provided for sklearn pipeline compatibility when working
        with single series. For detectors that require multi-series training,
        this method will raise an error - use fit() and transform() separately.

        Parameters
        ----------
        X : ArrayLike
            Time series data. Must be single series (n_samples, n_features).
            Lists of series are not supported in fit_transform().

        y : Segmentation | ArrayLike | None, default=None
            Optional labels for supervised learning.
            - Segmentation dict for changepoint labels
            - ArrayLike for dense per-sample labels
            - None for unsupervised learning

        **fit_params : dict
            Additional parameters passed to fit(). Not commonly used,
            but provided for sklearn compatibility.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Dense segment labels.

        Raises
        ------
        ValueError
            If X is a list of series. Multi-series training is not
            compatible with fit_transform - use fit() then transform().

        Examples
        --------
        >>> # Single series - works
        >>> detector = MyDetector()
        >>> labels = detector.fit_transform(X_train)

        >>> # Multi-series - raises error
        >>> detector.fit_transform([X1, X2, X3], y=[0, 1, 1])
        ValueError: fit_transform() does not support multi-series training...

        >>> # Multi-series - correct workflow
        >>> detector.fit([X1, X2, X3], y=[0, 1, 1])
        >>> labels = detector.transform(X_new)

        Notes
        -----
        The **fit_params argument exists for sklearn compatibility (e.g.,
        passing sample_weight through pipelines), but most change detectors
        won't use it.
        """
        if isinstance(X, list):
            raise ValueError(
                "fit_transform() does not support multi-series training. "
                "For detectors that require multiple series:\n"
                "  1. Train: detector.fit([X1, X2, X3], y=labels)\n"
                "  2. Transform: detector.transform(X_single)\n"
                "Use fit() and transform() separately instead of fit_transform()."
            )

        return self.fit(X, y, **fit_params).transform(X)

    def fit_predict(self, X, y=None, **fit_params):
        """Fit to data, then predict changepoints.

        This method is provided for sklearn compatibility when working
        with single series. For detectors that require multi-series training,
        this method will raise an error - use fit() and predict() separately.

        Parameters
        ----------
        X : ArrayLike
            Time series data. Must be single series (n_samples, n_features).
            Lists of series are not supported in fit_predict().

        y : Segmentation | ArrayLike | None, default=None
            Optional labels for supervised learning.
            - Segmentation dict for changepoint labels
            - ArrayLike for dense per-sample labels
            - None for unsupervised learning

        **fit_params : dict
            Additional parameters passed to fit(). Not commonly used,
            but provided for sklearn compatibility.

        Returns
        -------
        Segmentation
            Detection result as a dict with required fields:
            - "changepoints": np.ndarray, changepoint indices
            - "labels": np.ndarray, segment labels
            - "n_samples": int, number of samples

        Raises
        ------
        ValueError
            If X is a list of series. Multi-series training is not
            compatible with fit_predict - use fit() then predict().

        Examples
        --------
        >>> # Single series - works
        >>> detector = MyDetector()
        >>> result = detector.fit_predict(X_train)
        >>> print(result["changepoints"])

        >>> # Multi-series - raises error
        >>> detector.fit_predict([X1, X2, X3], y=[0, 1, 1])
        ValueError: fit_predict() does not support multi-series training...

        >>> # Multi-series - correct workflow
        >>> detector.fit([X1, X2, X3], y=[0, 1, 1])
        >>> result = detector.predict(X_new)

        Notes
        -----
        This is similar to fit_transform() but returns sparse Segmentation
        dict instead of dense labels. Use fit_transform() if you need
        dense labels for sklearn compatibility.
        """
        if isinstance(X, list):
            raise ValueError(
                "fit_predict() does not support multi-series training. "
                "For detectors that require multiple series:\n"
                "  1. Train: detector.fit([X1, X2, X3], y=labels)\n"
                "  2. Predict: detector.predict(X_single)\n"
                "Use fit() and predict() separately instead of fit_predict()."
            )

        return self.fit(X, y, **fit_params).predict(X)


# TODO: Finish tags.
# TODO: Add support for parameter specification. To simplify hyperparameter tuning
#       and validation.
