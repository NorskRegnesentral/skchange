"""Base classes for anomaly detectors."""

import numpy as np
import pandas as pd

from skchange.base import BaseDetector


class PointAnomalyDetector(BaseDetector):
    """Base class for anomaly detectors.

    Anomaly detectors detect individual data points that are considered anomalous.

    Output format of the predict method: See the dense_to_sparse method.
    Output format of the transform method: See the sparse_to_dense method.

    Subclasses should set the following tags for sktime compatibility:
    - task: "anomaly_detection"
    - learning_type: "unsupervised" or "supervised"
    - And possibly other tags, such as
        * "capability:missing_values": False,
        * "capability:multivariate": True,
        * "fit_is_empty": False,

    Needs to be implemented:
    - _fit(self, X, Y=None) -> self
    - _predict(self, X) -> pd.Series

    Optional to implement:
    - _score_transform(self, X) -> pd.Series
    - _update(self, X, Y=None) -> self
    """

    @staticmethod
    def sparse_to_dense(y_sparse: pd.Series, index: pd.Index) -> pd.Series:
        """Convert the sparse output from the predict method to a dense format.

        Parameters
        ----------
        y_sparse : pd.Series
            The sparse output from an anomaly detector's predict method.
        index : array-like
            Indices that are to be annotated according to ``y_sparse``.

        Returns
        -------
        pd.Series where 0-entries are normal and 1-entries are anomalous.
        """
        y_dense = pd.Series(0, index=index, name="anomaly_label", dtype="int64")
        y_dense.iloc[y_sparse.values] = 1
        return y_dense

    @staticmethod
    def dense_to_sparse(y_dense: pd.Series) -> pd.Series:
        """Convert the dense output from the transform method to a sparse format.

        Parameters
        ----------
        y_dense : pd.Series
            The dense output from an anomaly detector's transform method.
            0-entries are normal and >0-entries are anomalous.

        Returns
        -------
        pd.Series of the integer locations of the anomalous data points.
        """
        y_dense = y_dense.reset_index(drop=True)
        anomalies = y_dense.iloc[y_dense.values > 0].index
        return PointAnomalyDetector._format_sparse_output(anomalies)

    @staticmethod
    def _format_sparse_output(anomalies) -> pd.Series:
        """Format the sparse output of anomaly detectors.

        Can be reused by subclasses to format the output of the _predict method.
        """
        return pd.Series(anomalies, name="anomaly", dtype="int64")


class CollectiveAnomalyDetector(BaseDetector):
    """Base class for collective anomaly detectors.

    Collective anomaly detectors detect segments of data points that are considered
    anomalous.

    Output format of the predict method: See the dense_to_sparse method.
    Output format of the transform method: See the sparse_to_dense method.

    Subclasses should set the following tags for sktime compatibility:
    - task: "collective_anomaly_detection"
    - learning_type: "unsupervised" or "supervised"
    - And possibly other tags, such as
        * "capability:missing_values": False,
        * "capability:multivariate": True,
        * "fit_is_empty": False,

    Needs to be implemented:
    - _fit(self, X, Y=None) -> self
    - _predict(self, X) -> pd.Series

    Optional to implement:
    - _score_transform(self, X) -> pd.Series
    - _update(self, X, Y=None) -> self
    """

    @staticmethod
    def sparse_to_dense(y_sparse: pd.Series, index: pd.Index) -> pd.Series:
        """Convert the sparse output from the predict method to a dense format.

        Parameters
        ----------
        y_sparse : pd.Series[pd.Interval]
            The collective anomaly intervals.
        index : array-like
            Indices that are to be annotated according to ``y_sparse``.

        Returns
        -------
        pd.Series where 0-entries are normal and each collective anomaly are labelled
            from 1, ..., K.
        """
        labels = pd.IntervalIndex(y_sparse).get_indexer(index)
        # get_indexer return values 0 for the values inside the first interval, 1 to
        # the values within the next interval and so on, and -1 for values outside any
        # interval. The skchange convention is that 0 is normal and > 0 is anomalous,
        # so we add 1 to the result.
        labels += 1
        return pd.Series(labels, index=index, name="anomaly_label", dtype="int64")

    @staticmethod
    def dense_to_sparse(y_dense: pd.Series) -> pd.Series:
        """Convert the dense output from the transform method to a sparse format.

        Parameters
        ----------
        y_dense : pd.Series
            The dense output from a collective anomaly detector's transform method:
            An integer series where 0-entries are normal and each collective anomaly
            are labelled from 1, ..., K.

        Returns
        -------
        pd.Series[pd.Interval] containing the collective anomaly intervals.

        Notes
        -----
        The start and end points of the intervals can be accessed by
        output.array.left and output.array.right, respectively.
        """
        y_dense = y_dense.reset_index(drop=True)
        y_anomaly = y_dense.loc[y_dense.values > 0]
        anomaly_locations_diff = y_anomaly.index.diff()

        first_anomaly_start = y_anomaly.index[:1].to_numpy()
        anomaly_starts = y_anomaly.index[anomaly_locations_diff > 1]
        anomaly_starts = np.insert(anomaly_starts, 0, first_anomaly_start)

        last_anomaly_end = y_anomaly.index[-1:].to_numpy()
        anomaly_ends = y_anomaly.index[np.roll(anomaly_locations_diff > 1, -1)]
        anomaly_ends = np.insert(anomaly_ends, len(anomaly_ends), last_anomaly_end)

        anomaly_intervals = list(zip(anomaly_starts, anomaly_ends))
        return CollectiveAnomalyDetector._format_sparse_output(anomaly_intervals)

    @staticmethod
    def _format_sparse_output(anomaly_intervals, closed="both") -> pd.Series:
        """Format the sparse output of collective anomaly detectors.

        Can be reused by subclasses to format the output of the _predict method.
        """
        return pd.Series(
            pd.IntervalIndex.from_tuples(anomaly_intervals, closed=closed),
            name="collective_anomaly",
        )


class SubsetCollectiveAnomalyDetector(BaseDetector):
    """Base class for subset collective anomaly detectors.

    Subset collective anomaly detectors detect segments of multivariate time series data
    that are considered anomalous, and also provide information on which components of
    the data are affected.

    Output format of the predict method:
    pd.DataFrame({
        "location": pd.IntervalIndex(anomaly_intervals, closed=<insert>),
        "columns": affected_components_list,
    })

    Subclasses should set the following tags for sktime compatibility:
    - task: "collective_anomaly_detection"
    - learning_type: "unsupervised" or "supervised"
    - capability:subset_detection: True
    - And possibly other tags, such as
        * "capability:missing_values": False,
        * "capability:multivariate": True,
        * "fit_is_empty": False,

    Needs to be implemented:
    - _fit(self, X, Y=None) -> self
    - _predict(self, X) -> pd.DataFrame

    Optional to implement:
    - _score_transform(self, X) -> pd.Series
    - _update(self, X, Y=None) -> self
    """

    @staticmethod
    def sparse_to_dense(y_sparse, index):
        """Convert the sparse output from the predict method to a dense format.

        Parameters
        ----------
        y_sparse : pd.DataFrame
            The sparse output from the predict method.
        index : array-like
            Indices that are to be annotated according to ``y_sparse``.

        Returns
        -------
        pd.DataFrame
        """
