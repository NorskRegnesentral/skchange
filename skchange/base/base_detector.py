"""Detector base class.

    class name: BaseDetector

    Adapted from the BaseSeriesAnnotator class in sktime.

Scitype defining methods:
    fitting                         - fit(self, X, y=None)
    detecting, sparse format        - predict(self, X)
    detecting, dense format         - transform(self, X)
    detection scores, dense         - transform_scores(self, X)  [optional]
    updating (temporal)             - update(self, X, y=None)  [optional]

Each detector type (e.g. point anomaly detector, collective anomaly detector,
changepoint detector) are subclasses of BaseDetector (task tag in sktime).
A detector type is defined by the content and format of the output of the predict
method. Each detector type therefore has the following methods for converting between
sparse and dense output formats:
    converting sparse output to dense - sparse_to_dense(y_sparse, index, columns)
    converting dense output to sparse - dense_to_sparse(y_dense)  [optional]

Convenience methods:
    update&detect   - update_predict(self, X)
    fit&detect      - fit_predict(self, X, y=None)
    fit&transform   - fit_transform(self, X, y=None)

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - check_is_fitted()

Needs to be implemented for a concrete detector:
    _fit(self, X, y=None)
    _predict(self, X)
    sparse_to_dense(y_sparse, index)

Recommended but optional to implement for a concrete detector:
    dense_to_sparse(y_dense)
    _transform_scores(self, X)
    _update(self, X, y=None)
"""

__author__ = ["Tveten"]
__all__ = ["BaseDetector"]

import pandas as pd
from sktime.base import BaseEstimator
from sktime.utils.validation.series import check_series


class BaseDetector(BaseEstimator):
    """Base class for all detectors in skchange.

    A detector is a model that detects events in time series data. Events can be
    anomalies, changepoints, or any other type of event that the detector is designed
    to detect. The output from a detector is a series of detections, where each element
    corresponds to a detected event. The format of the output depends on the detector
    type.

    The base detector class defines the interface for all detectors in skchange. It
    defines the methods that all detectors should implement, as well as some optional
    methods that can be implemented if they are relevant for a specific detector type.

    The `predict` method returns the detections in a sparse format, where each element
    corresponds to a detected event. The `transform` method returns the detections in
    a dense format, where each element in the input data is annotated according to the
    detection results. The `transform_scores` method (if implemented) returns detection
    scores in a dense format.

    In addition, there are two special format defining and converting methods that
    should be implemented for each detector type: `sparse_to_dense` and
    `dense_to_sparse`. These methods define the format of the output from the detector
    and how to convert between the sparse and dense formats. The `transform` method
    uses `sparse_to_dense` to convert the output from `predict` to a dense format.
    It will not work if `sparse_to_dense` is not implemented.

    Note that the `BaseDetector` serves as an alternative to the `BaseSeriesAnnotator`
    class in sktime, specifically tailored for detection-oriented tasks.
    """

    _tags = {
        # "object_type": "detector",  # type of object
        "authors": "Tveten",  # author(s) of the object
        "maintainers": "Tveten",  # current maintainer(s) of the object
        "distribution_type": None,
    }  # for unit test cases

    def __init__(self):
        self.task = self.get_class_tag("task")
        self.learning_type = self.get_class_tag("learning_type")

        self._is_fitted = False

        self._X = None
        self._y = None

        super().__init__()

    def fit(self, X, y=None):
        """Fit detector to training data.

        Fit trains the detector on the input data, for example by tuning a detection
        threshold or other hyperparameters. Detection of events does not happen here,
        but in the `predict` or `transform` methods, after the detector has been
        fit.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame or np.ndarray
            Training data to fit model to (time series).
        y : pd.Series, pd.DataFrame or np.ndarray, optional.
            Ground truth detections for training if detector is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Creates fitted model that updates attributes ending in "_". Sets
        `_is_fitted` flag to True.
        """
        X = check_series(X, allow_index_names=True)

        if y is not None:
            y = check_series(y, allow_index_names=True)

        self._X = X
        self._y = y

        # fkiraly: insert checks/conversions here, after PR #1012 I suggest

        self._fit(X=X, y=y)

        # this should happen last
        self._is_fitted = True

        return self

    def _fit(self, X, y=None):
        """Fit detector to training data.

        The core logic for fitting the detector to training data should be implemented
        here. A typical example is to tune a detection threshold to training data.
        This method must be implemented by all detectors.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame or np.ndarray
            Training data to fit model to (time series).
        y : pd.Series, pd.DataFrame or np.ndarray, optional
            Ground truth detections for training if detector is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        raise NotImplementedError("abstract method")

    def predict(self, X):
        """Detect events and return the result in a sparse format.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame or np.ndarray
            Data to detect events in (time series).

        Returns
        -------
        y : pd.DataFrame
            Each element or row corresponds to a detected event. Exact format depends on
            the detector type.
        """
        self.check_is_fitted()

        X = check_series(X, allow_index_names=True)

        # fkiraly: insert checks/conversions here, after PR #1012 I suggest

        y = self._predict(X=X)
        return y

    def _predict(self, X):
        """Detect events and return the result in a sparse format.

        The core logic for detecting events in the input data should be implemented
        here. This method must be implemented by all detectors.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame or np.ndarray
            Data to detect events in (time series).

        Returns
        -------
        y : pd.DataFrame
            Each element or row corresponds to a detected event. Exact format depends on
            the detector type.

        y : pd.DataFrame with RangeIndex
            Detected or predicted events.

            Each (axis 0) index of ``y`` is a detected event.

            Has the following columns:

            * ``"ilocs"`` - always. Values are ``iloc`` references to indices of ``X``,
            signifying the integer location of the detected event in ``X``.
            * ``"label"`` - optional, additional label information.

            The meaning of entries in the ``"ilocs"`` column and ``"labels"``
            column is as follows:
            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              ``"ilocs"`` contains the iloc index of the event, and
              labels (if present) signify types of events.
            * If ``task`` is ``"segmentation"``, ``"ilocs"`` contains left-closed
              intervals of iloc based segments, and labels (if present)
              are types of segments.
        """
        raise NotImplementedError("abstract method")

    def transform(self, X):
        """Detect events and return the result in a dense format.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame or np.ndarray
            Data to detect events in (time series).

        Returns
        -------
        y : pd.DataFrame
            If the detector does not have the capability to identify subsets of
            variables that are affected by the detected events, out output will be a
            `pd.DataFrame` with the same index as X and one column:

            * `"labels"`: Integer labels starting from 0.

            If the detector has this capability, the output will be a `pd.DataFrame`
            with the same index as X and as many columns as there are columns in X
            of the following format:

            * `"labels_<X.columns[i]>"` for each column index i in X.columns: Integer
            labels starting from 0.
        """
        y = self.predict(X)
        X_inner = pd.DataFrame(X)
        y_dense = self.sparse_to_dense(y, X_inner.index, X_inner.columns)
        return y_dense

    @staticmethod
    def sparse_to_dense(y_sparse, index, columns=None):
        """Convert the sparse output from a detector to a dense format.

        Parameters
        ----------
        y_sparse : pd.Series
            The sparse output from a detector's `predict` method. The format of the
            series depends on the task and capability of the annotator.
        index : array-like
            Indices that are to be annotated according to `y_sparse`.
        columns : array-like, optional
            Columns that are to be annotated according to `y_sparse`.

        Returns
        -------
        pd.Series or pd.DataFrame of detection labels.
        """
        raise NotImplementedError("abstract method")

    @staticmethod
    def dense_to_sparse(y_dense):
        """Convert the dense output from a detector to a sparse format.

        Parameters
        ----------
        y_dense : pd.Series
            The dense output from a detector's `transform` method. The format of the
            series depends on the task and capability of the annotator.

        Returns
        -------
        pd.Series
        """
        raise NotImplementedError("abstract method")

    def transform_scores(self, X):
        """Return detection scores on the input data.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame or np.ndarray
            Data to score (time series).

        Returns
        -------
        y : pd.Series or pd.DataFrame
            Scores for sequence X. Exact format depends on the concrete detector.
        """
        self.check_is_fitted()
        X = check_series(X, allow_index_names=True)
        return self._transform_scores(X)

    def _transform_scores(self, X):
        """Return detection scores on the input data.

        The core logic for scoring the input data should be implemented here. This
        method is optional to implement, but is useful for detectors that provide
        detection scores.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame or np.ndarray
            Data to score (time series).

        Returns
        -------
        y : pd.Series or pd.DataFrame
            Scores for sequence X. Exact format depends on the concrete detector.
        """
        raise NotImplementedError("abstract method")

    def update(self, X, y=None):
        """Update model with new data and optional ground truth detections.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame or np.ndarray
            Training data to update model with (time series).
        y : pd.Series, pd.DataFrame or np.ndarray, optional
            Ground truth detections for training if detector is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        self.check_is_fitted()

        X = check_series(X, allow_index_names=True)

        if y is not None:
            y = check_series(y, allow_index_names=True)

        self._X = X.combine_first(self._X)

        if y is not None:
            self._y = y.combine_first(self._y)

        self._update(X=X, y=y)

        return self

    def _update(self, X, y=None):
        """Update model with new data and optional ground truth detections.

        The core logic for updating the detector with new data should be implemented
        here. This method is optional to implement, but is useful for detectors that
        can be efficiently updated with new data.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame or np.ndarray
            Training data to update model with (time series).
        y : pd.Series, pd.DataFrame or np.ndarray, optional
            Ground truth detections for training if detector is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        # default/fallback: re-fit to all data
        self._fit(self._X, self._y)

        return self

    def update_predict(self, X, y=None):
        """Update model with new data and detect events in it.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame or np.ndarray
            Training data to update model with and detect events in (time series).
        y : pd.Series, pd.DataFrame or np.ndarray, optional
            Ground truth detections for training if detector is supervised.

        Returns
        -------
        y : pd.Series or pd.DataFrame
            Each element or row corresponds to a detected event. Exact format depends on
            the detector type.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        self.update(X=X)
        y = self.predict(X=X)

        return y

    def fit_predict(self, X, y=None):
        """Fit to data, then predict it.

        Fits model to `X` and `y` with given detector parameters and returns the
        detected events.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Training data to fit model with and detect events in (time series).
        y : pd.Series, pd.DataFrame or np.ndarray, optional
            Ground truth detections for training if detector is supervised.

        Returns
        -------
        y : pd.Series or pd.DataFrame
            Each element or row corresponds to a detected event. Exact format depends on
            the detector type.
        """
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        return self.fit(X, y).predict(X)

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits model to `X` and `y` with given detector parameters and returns the
        detected events in a dense format.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Training data to fit model with and detect events in (time series).
        y : pd.Series or np.ndarray, optional (default=None)
            Target values of data to be detected.

        Returns
        -------
        y : pd.Series or pd.DataFrame
            Detections for sequence `X`. The returned detections will be in the dense
            format, meaning that each element in `X` will be annotated according to the
            detection results in some meaningful way depending on the detector type.
        """
        return self.fit(X, y).transform(X)
