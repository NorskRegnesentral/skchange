"""Base detector implementation providing sklearn compatibility.

Provides a minimal base class for change detectors that:
- Inherits from sklearn.base.BaseEstimator for compatibility
- Defines custom tags via __sklearn_tags__()
- Provides default predict() derived from predict_changepoints()

Subclasses must implement fit() and predict_changepoints().
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import get_tags

from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._conversion import changepoints_to_labels
from skchange.new_api.utils._tags import ChangeDetectorTags, SkchangeTags


class BaseChangeDetector(BaseEstimator):
    """Base class for change detectors providing sklearn compatibility.

    Inherits from sklearn.base.BaseEstimator for sklearn compatibility,
    including cloning, pipeline support, and get_params/set_params.

    This base class provides:
    - Custom tags via __sklearn_tags__() for change detection metadata
    - Default predict() returning dense (n_samples,) segment labels
    - Default fit_predict() convenience method

    Subclasses must implement:
    - fit(X, y=None) -> self
    - predict_changepoints(X) -> np.ndarray of changepoint indices

    predict() is derived automatically from predict_changepoints().
    Subclasses may override predict() directly when the algorithm natively
    produces dense labels.

    Examples
    --------
    >>> class MyDetector(BaseChangeDetector):
    ...     def __init__(self, threshold=1.0):
    ...         self.threshold = threshold
    ...
    ...     def fit(self, X, y=None):
    ...         self.n_features_in_ = X.shape[1]
    ...         return self
    ...
    ...     def predict_changepoints(self, X):
    ...         return np.array([50, 100])
    """

    def predict_changepoints(self, X: ArrayLike) -> np.ndarray:
        """Detect changepoints in a time series.

        Subclasses must implement this method.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series data.

        Returns
        -------
        changepoints : np.ndarray of shape (n_changepoints,)
            Indices where structural breaks occur. Empty array if none detected.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement predict_changepoints()."
        )

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Detect changepoints, returning per-sample segment labels.

        Default implementation calls predict_changepoints() and converts the
        changepoint indices to an array with segment labels per sample. Subclasses may
        override this directly when the algorithm natively produces labels.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series data.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Dense integer segment labels, one per sample. Segment 0 is the
            first segment, segment 1 the next, and so on.

        Examples
        --------
        >>> labels = detector.fit(X_train).predict(X_test)
        >>> labels.shape
        (n_samples,)
        """
        changepoints = self.predict_changepoints(X)
        return changepoints_to_labels(changepoints, n_samples=len(X))

    def fit_predict(self, X, y: ArrayLike | None = None, **fit_params) -> np.ndarray:
        """Fit to data, then predict changepoint indices.

        Equivalent to calling fit(X, y).predict_changepoints(X).

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series data.

        y : None
            Ignored. Exists for sklearn API compatibility.

        **fit_params : dict
            Additional parameters passed to fit().

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Dense integer segment labels, one per sample. Segment 0 is the
            first segment, segment 1 the next, and so on.
        """
        return self.fit(X, y, **fit_params).predict(X)

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
        return tags


def is_change_detector(estimator) -> bool:
    """Return True if the given estimator is (probably) a change detector.

    Parameters
    ----------
    estimator : estimator instance
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a change detector and False otherwise.

    Examples
    --------
    >>> from skchange.new_api.detectors import MovingWindow, is_change_detector
    >>> is_change_detector(MovingWindow())
    True
    """
    return get_tags(estimator).change_detector_tags is not None  # type: ignore[union-attr]
