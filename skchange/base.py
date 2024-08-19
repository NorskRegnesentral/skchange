"""Detector base class.

    class name: BaseSeriesAnnotator

Scitype defining methods:
    fitting              - fit(self, X, Y=None)
    detecting            - predict(self, X)
    updating (temporal)  - update(self, X, Y=None)
    update&detect        - update_predict(self, X)

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - check_is_fitted()

"""

__author__ = ["mtveten"]
__all__ = ["BaseDetector"]

from sktime.base import BaseEstimator
from sktime.utils.validation.series import check_series


class BaseDetector(BaseEstimator):
    """Base detector.

    An alternative implementation to the BaseSeriesAnnotator class from sktime,
    more focused on the detection of events of interest.
    Enables quicker bug fixes for example, since the annotation module is still
    experimental.

    All detectors share the common feature that each element of the output from .predict
    indicates the detection of a specific event of interest, such as an anomaly, a
    changepoint, or something else.

    Needs to be implemented:
    - _fit(self, X, Y=None) -> self
    - _predict(self, X) -> pd.Series or pd.DataFrame

    Optional to implement:
    - _transform_scores(self, X) -> pd.Series or pd.DataFrame
    - _update(self, X, Y=None) -> self

    Required .predict output formats per task and capability:
    - task == "anomaly_detection":
        pd.Series(anomaly_indices, dtype=int, name="anomalies)
    - task == "collective_anomaly_detection":
        pd.Series(pd.IntervalIndex(
            anomaly_intervals, closed=<insert>, name="collective_anomalies"
        ))
    - task == "change_point_detection":
        Changepoints are defined as the last element of a segment.
        pd.Series(changepoint_indices, dtype=int, name="changepoints")
    - task == "segmentation":
        Difference from change point detection: Allows the same label to be assigned to
        multiple segments.
        pd.Series({
            index = pd.IntervalIndex(segment_intervals, closed=<insert>),
            values = segment_labels,
        })
    - task == "None":
        Custom task.
        Only restriction is that the output must be a pd.Series or pd.DataFrame where
        each element or row corresponds to a detected event.
        For .transform to work, .sparse_to_dense must be implemented for custom tasks.
    - capability:subset_detection is True:
        * task == "anomaly_detection":
            pd.DataFrame({
                "location": anomaly_indices,
                "columns": affected_components_list,
            })
        * task == "collective_anomaly_detection":
            pd.DataFrame({
                "location": pd.IntervalIndex(anomaly_intervals, closed=<insert>),
                "columns": affected_components_list,
            })
        * task == "change_point_detection":
            pd.DataFrame({
                "location": changepoint_indices,
                "columns": affected_components_list,
            })
    - capability:detection_score is True: Explicit way of stating that _transform_scores
      is implemented.
    """

    _tags = {
        "object_type": "detector",  # type of object
        "learning_type": "None",  # Tag to determine test in test_all_annotators
        "task": "None",  # Tag to determine test in test_all_annotators
        #
        # todo: distribution_type? we may have to refactor this, seems very soecufuc
        "distribution_type": "None",  # Tag to determine test in test_all_annotators
    }  # for unit test cases

    def __init__(self):
        self.task = self.get_class_tag("task")
        self.learning_type = self.get_class_tag("learning_type")

        self._is_fitted = False

        self._X = None
        self._Y = None

        super().__init__()

    def _fit(self, X, Y=None):
        """Fit to training data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit model to time series.
        Y : pd.Series, optional
            Ground truth annotations for training if annotator is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        raise NotImplementedError("abstract method")

    def _predict(self, X):
        """Create annotations on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Data to annotate, time series.

        Returns
        -------
        Y : pd.Series
            Annotations for sequence X exact format depends on annotation type.
        """
        raise NotImplementedError("abstract method")

    def _transform_scores(self, X):
        """Return scores for predicted annotations on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Data to annotate, time series.

        Returns
        -------
        Y : pd.Series
            One score for each element in X.
            Annotations for sequence X exact format depends on annotation type.
        """
        raise NotImplementedError("abstract method")

    def _update(self, X, Y=None):
        """Update model with new data and optional ground truth annotations.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Training data to update model with time series
        Y : pd.Series, optional
            Ground truth annotations for training if annotator is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        # default/fallback: re-fit to all data
        self._fit(self._X, self._Y)

        return self

    def fit(self, X, Y=None):
        """Fit to training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit model to (time series).
        Y : pd.Series, optional
            Ground truth annotations for training if annotator is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Creates fitted model that updates attributes ending in "_". Sets
        _is_fitted flag to True.
        """
        X = check_series(X)

        if Y is not None:
            Y = check_series(Y)

        self._X = X
        self._Y = Y

        # fkiraly: insert checks/conversions here, after PR #1012 I suggest

        self._fit(X=X, Y=Y)

        # this should happen last
        self._is_fitted = True

        return self

    def predict(self, X):
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to annotate (time series).

        Returns
        -------
        Y : pd.Series
            Annotations for sequence X exact format depends on annotation type.
        """
        self.check_is_fitted()

        X = check_series(X)

        # fkiraly: insert checks/conversions here, after PR #1012 I suggest

        Y = self._predict(X=X)

        return Y

    def transform(self, X):
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to annotate (time series).

        Returns
        -------
        Y : pd.Series
            Annotations for sequence X. The returned annotations will be in the dense
            format.
        """
        Y = self.predict(X)
        return self.sparse_to_dense(Y, X.index)

    def sparse_to_dense(self, y_sparse, index):
        """Convert the sparse output from an annotator to a dense format.

        Parameters
        ----------
        y_sparse : pd.Series
            The sparse output from an annotator's predict method. The format of the
            series depends on the task and capability of the annotator.
        index : array-like
            Indices that are to be annotated according to ``y_sparse``.

        Returns
        -------
        pd.Series
        """
        if self.get_class_tag("capability:subset_detection"):
            y_sparse = y_sparse["location"]

        if self.task == "segmentation":
            return self.sparse_to_dense_segmentation(y_sparse, index)
        elif self.task == "change_point_detection":
            return self.sparse_to_dense_change_points(y_sparse, index)
        elif self.task == "anomaly_detection":
            return self.sparse_to_dense_anomalies(y_sparse, index)
        elif self.task == "collective_anomaly_detection":
            return self.sparse_to_dense_collective_anomalies(y_sparse, index)
        else:
            # Overwrite sparse_to_dense for custom tasks.
            raise NotImplementedError(
                f"sparse_to_dense not implemented for task='{self.task}'"
            )

    @staticmethod
    def sparse_to_dense_segmentation(y_sparse, index):
        """Convert the output from a segmentation annotator to a dense format."""

    @staticmethod
    def sparse_to_dense_change_points(y_sparse, index):
        """Convert the output from a change point detector to a dense format."""

    @staticmethod
    def sparse_to_dense_anomalies(y_sparse, index):
        """Convert the output from an anomaly detector to a dense format."""

    @staticmethod
    def sparse_to_dense_collective_anomalies(y_sparse, index):
        """Convert the output from a collective anomaly detector to a dense format."""

    def transform_scores(self, X):
        """Return scores for predicted annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to annotate (time series).

        Returns
        -------
        Y : pd.Series
            Scores for sequence X exact format depends on annotation type.
        """
        self.check_is_fitted()
        X = check_series(X)
        return self._transform_scores(X)

    def update(self, X, Y=None):
        """Update model with new data and optional ground truth annotations.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to update model with (time series).
        Y : pd.Series, optional
            Ground truth annotations for training if annotator is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        self.check_is_fitted()

        X = check_series(X)

        if Y is not None:
            Y = check_series(Y)

        self._X = X.combine_first(self._X)

        if Y is not None:
            self._Y = Y.combine_first(self._Y)

        self._update(X=X, Y=Y)

        return self

    def update_predict(self, X):
        """Update model with new data and create annotations for it.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to update model with, time series.

        Returns
        -------
        Y : pd.Series
            Annotations for sequence X exact format depends on annotation type.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        X = check_series(X)

        self.update(X=X)
        Y = self.predict(X=X)

        return Y

    def fit_predict(self, X, Y=None):
        """Fit to data, then predict it.

        Fits model to X and Y with given annotation parameters
        and returns the annotations made by the model.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to be transformed
        Y : pd.Series or np.ndarray, optional (default=None)
            Target values of data to be predicted.

        Returns
        -------
        self : pd.Series
            Annotations for sequence X exact format depends on annotation type.
        """
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        return self.fit(X, Y).predict(X)

    def fit_transform(self, X, Y=None):
        """Fit to data, then transform it.

        Fits model to X and Y with given annotation parameters
        and returns the annotations made by the model.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to be transformed
        Y : pd.Series or np.ndarray, optional (default=None)
            Target values of data to be predicted.

        Returns
        -------
        self : pd.Series
            Annotations for sequence X exact format depends on annotation type.
        """
        Y = self.fit_predict(X)
        return self.sparse_to_dense(Y, index=X.index)

    # def predict_segments(self, X):
    #     """Predict segments on test/deployment data.

    #     Parameters
    #     ----------
    #     X : pd.DataFrame
    #         Data to annotate, time series.

    #     Returns
    #     -------
    #     Y : pd.Series
    #         A series with an index of intervals. Each interval is the range of a
    #         segment and the corresponding value is the label of the segment.
    #     """
    #     self.check_is_fitted()
    #     X = check_series(X)

    #     predict_output = self.predict(X)
    #     if self.get_class_tag("capability:subset_detection"):
    #         predict_output = predict_output["location"]

    #     if self.task == "segmentation":
    #         return predict_output
    #     elif self.task == "change_point_detection":
    #         return self.change_points_to_segments(
    #             predict_output, start=X.index.min(), end=X.index.max()
    #         )
    #     elif self.task == "anomaly_detection":
    #         return self.point_anomalies_to_segments(
    #             predict_output, start=X.index.min(), end=X.index.max()
    #         )
    #     elif self.task == "collective_anomaly_detection":
    #         return self.collective_anomalies_to_segments(
    #             predict_output, start=X.index.min(), end=X.index.max()
    #         )

    # def predict_points(self, X):
    #     """Predict changepoints/anomalies on test/deployment data.

    #     Parameters
    #     ----------
    #     X : pd.DataFrame
    #         Data to annotate, time series.

    #     Returns
    #     -------
    #     Y : pd.Series
    #         A series whose values are the changepoints/anomalies in X.
    #     """
    #     self.check_is_fitted()
    #     X = check_series(X)

    #     predict_output = self.predict(X)
    #     if self.get_class_tag("capability:subset_detection"):
    #         predict_output = predict_output["location"]

    #     if self.task == "anomaly_detection" or self.task == "change_point_detection":
    #         return predict_output
    #     elif self.task == "collective_anomaly_detection":
    #         # TODO Add support. Turn collective anomalies into point anomalies.
    #         return self.collective_anomalies_to_point_anomalies(predict_output)
    #     elif self.task == "segmentation":
    #         return self.segments_to_change_points(predict_output)

    # @staticmethod
    # def point_anomalies_to_segments(self, anomalies, start, end):
    #     # TODO Add support. 0 = normal, 1, ..., K = anomaly.
    #     pass

    # @staticmethod
    # def collective_anomalies_to_segments(self, collective_anomalies, start, end):
    #     # TODO Add support. 0 = normal, 1, ..., K = anomaly.
    #     pass

    # @staticmethod
    # def collective_anomalies_to_point_anomalies(self, collective_anomalies):
    #     pass

    # @staticmethod
    # def change_points_to_segments(y_sparse, start, end):
    #     """Convert a series of change point indexes to segments.

    #     Parameters
    #     ----------
    #     y_sparse : pd.Series
    #         A series containing the indexes of change points.
    #     start : optional
    #         Starting point of the first segment.
    #     end : optional
    #         Ending point of the last segment

    #     Returns
    #     -------
    #     pd.Series
    #         A series with an interval index indicating the start and end points of the
    #         segments. The values of the series are the labels of the segments.

    #     Examples
    #     --------
    #     >>> import pandas as pd
    #     >>> from sktime.annotation.base._base import BaseSeriesAnnotator
    #     >>> change_points = pd.Series([1, 2, 5])
    #     >>> BaseSeriesAnnotator.change_points_to_segments(change_points, 0, 7)
    #     [0, 1)   -1
    #     [1, 2)    1
    #     [2, 5)    2
    #     [5, 7)    3
    #     dtype: int64
    #     """
    #     breaks = y_sparse.values

    #     if start > breaks.min():
    #         raise ValueError(
    #             "The starting index must be before the first change point."
    #         )
    #     first_change_point = breaks.min()

    #     if start is not None:
    #         breaks = np.insert(breaks, 0, start)
    #     if end is not None:
    #         breaks = np.append(breaks, end)

    #     index = pd.IntervalIndex.from_breaks(breaks, copy=True, closed="left")
    #     segments = pd.Series(0, index=index)

    #     in_range = index.left >= first_change_point

    #     number_of_segments = in_range.sum()
    #     segments.loc[in_range] = range(1, number_of_segments + 1)
    #     segments.loc[~in_range] = -1

    #     return segments
