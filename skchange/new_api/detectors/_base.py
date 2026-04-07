"""Base class for all change and anomaly detectors in the new API."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import get_tags

from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._conversion import changepoints_to_labels
from skchange.new_api.utils._tags import ChangeDetectorTags, SkchangeTags


class BaseChangeDetector(BaseEstimator):
    """Base class for all detectors providing sklearn compatibility.

    Inherits from ``sklearn.base.BaseEstimator`` for cloning, pipeline support,
    and ``get_params`` / ``set_params``.

    Subclasses must implement:

    - ``fit(X, y=None) -> self``
    - ``predict_changepoints(X) -> np.ndarray`` of sorted boundary indices

    Both ``predict`` and ``predict_changepoints`` are the universal interface:
    every detector supports both. ``predict_changepoints`` returns sorted
    boundary indices; ``predict`` returns one segment label per input sample
    (labels are integers and may reoccur across non-contiguous segments).

    ``predict`` is derived from ``predict_changepoints`` by default. Subclasses
    that natively produce labels (e.g. CAPA uses ``0 = normal, 1..K = anomaly``)
    override ``predict`` directly and also override ``predict_changepoints``
    to derive boundaries from the labels.

    Additional capabilities such as ``predict_all()`` or
    ``predict_segment_anomalies()`` are duck-typed: add them on the concrete
    class when the algorithm supports them. No intermediate base class is
    needed.

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
            Sorted integer indices of detected changepoints. A changepoint at
            index ``t`` means sample ``t`` is the first sample of a new segment,
            i.e. a structural break occurs between samples ``t-1`` and ``t``.
            Empty array if no changepoints are detected.
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
