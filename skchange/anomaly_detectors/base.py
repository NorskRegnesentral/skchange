"""Base classes for anomaly detectors.

    classes:
        PointAnomalyDetector
        CollectiveAnomalyDetector
        SubsetCollectiveAnomalyDetector

By inheriting from these classes the remaining methods of the BaseDetector class to
implement to obtain a fully functional anomaly detector are given below.

Needs to be implemented:
    _fit(self, X, y=None)
    _predict(self, X)

Optional to implement:
    _score_transform(self, X)
    _update(self, X, y=None)
"""

import numpy as np
import pandas as pd

from skchange.base import BaseDetector


class PointAnomalyDetector(BaseDetector):
    """Base class for point anomaly detectors.

    Point anomaly detectors detect individual data points that are considered anomalous.

    Output format of the predict method: See the dense_to_sparse method.
    Output format of the transform method: See the sparse_to_dense method.
    """

    @staticmethod
    def sparse_to_dense(
        y_sparse: pd.Series, index: pd.Index, columns: pd.Index = None
    ) -> pd.Series:
        """Convert the sparse output from the predict method to a dense format.

        Parameters
        ----------
        y_sparse : pd.Series
            The sparse output from an anomaly detector's predict method.
        index : array-like
            Indices that are to be annotated according to ``y_sparse``.
        columns: array-like
            Not used. Only for API compatibility.

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
        # The sparse format only uses integer positions, so we reset the index.
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

    Output format of the `predict` method: See the `dense_to_sparse` method.
    Output format of the `transform` method: See the `sparse_to_dense` method.
    """

    @staticmethod
    def sparse_to_dense(
        y_sparse: pd.Series, index: pd.Index, columns: pd.Index = None
    ) -> pd.Series:
        """Convert the sparse output from the `predict` method to a dense format.

        Parameters
        ----------
        y_sparse : pd.Series[pd.Interval]
            The collective anomaly intervals.
        index : array-like
            Indices that are to be annotated according to `y_sparse`.
        columns: array-like
            Not used. Only for API compatibility.

        Returns
        -------
        pd.Series where 0-entries are normal and each collective anomaly are labelled
            from 1, ..., K.
        """
        labels = pd.IntervalIndex(y_sparse).get_indexer(index)
        # `get_indexer` return values 0 for the values inside the first interval, 1 to
        # the values within the next interval and so on, and -1 for values outside any
        # interval. The `skchange` convention is that 0 is normal and > 0 is anomalous,
        # so we add 1 to the result.
        labels += 1
        return pd.Series(labels, index=index, name="anomaly_label", dtype="int64")

    @staticmethod
    def dense_to_sparse(y_dense: pd.Series) -> pd.Series:
        """Convert the dense output from the `transform` method to a sparse format.

        Parameters
        ----------
        y_dense : pd.Series
            The dense output from a collective anomaly detector's `transform` method:
            An integer series where 0-entries are normal and each collective anomaly
            are labelled from 1, ..., K.

        Returns
        -------
        pd.Series[pd.Interval] containing the collective anomaly intervals.

        Notes
        -----
        The start and end points of the intervals can be accessed by
        `output.array.left` and `output.array.right`, respectively.
        """
        # The sparse format only uses integer positions, so we reset the index.
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
        return CollectiveAnomalyDetector._format_sparse_output(
            anomaly_intervals, closed="both"
        )

    @staticmethod
    def _format_sparse_output(
        anomaly_intervals: list[tuple[int, int]], closed: str = "both"
    ) -> pd.Series:
        """Format the sparse output of collective anomaly detectors.

        Can be reused by subclasses to format the output of the `_predict` method.
        """
        return pd.Series(
            pd.IntervalIndex.from_tuples(anomaly_intervals, closed=closed),
            name="anomaly_interval",
        )


class SubsetCollectiveAnomalyDetector(BaseDetector):
    """Base class for subset collective anomaly detectors.

    Subset collective anomaly detectors detect segments of multivariate time series data
    that are considered anomalous, and also provide information on which components of
    the data are affected.

    Output format of the `predict` method: See the `dense_to_sparse` method.
    Output format of the `transform` method: See the `sparse_to_dense` method.
    """

    @staticmethod
    def sparse_to_dense(
        y_sparse: pd.DataFrame, index: pd.Index, columns: pd.Index
    ) -> pd.DataFrame:
        """Convert the sparse output from the `predict` method to a dense format.

        Parameters
        ----------
        y_sparse : pd.DataFrame
            The sparse output from the `predict` method. The first column must contain
            the anomaly intervals, the second column must contain a list of the affected
            columns.
        index : array-like
            Indices that are to be annotated according to `y_sparse`.
        columns : array-like
            Columns that are to be annotated according to `y_sparse`.

        Returns
        -------
        pd.DataFrame where 0-entries are normal and each collective anomaly are labelled
            from 1, ..., K.
        """
        anomaly_intervals = y_sparse.iloc[:, 0].array
        anomaly_starts = anomaly_intervals.left
        anomaly_ends = anomaly_intervals.right
        anomaly_columns = y_sparse.iloc[:, 1]

        start_is_open = anomaly_intervals.closed in ["neither", "right"]
        if start_is_open:
            anomaly_starts += 1  # Exclude the start index in the for loop below.
        end_is_closed = anomaly_intervals.closed in ["both", "right"]
        if end_is_closed:
            anomaly_ends += 1  # Include the end index in the for loop below.

        labels = np.zeros((len(index), len(columns)), dtype="int64")
        anomalies = zip(anomaly_starts, anomaly_ends, anomaly_columns)
        for i, (start, end, affected_columns) in enumerate(anomalies):
            labels[start:end, affected_columns] = i + 1

        return pd.DataFrame(labels, index=index, columns=columns)

    @staticmethod
    def dense_to_sparse(y_dense: pd.DataFrame):
        """Convert the dense output from the `transform` method to a sparse format.

        Parameters
        ----------
        y_dense : pd.DataFrame
            The dense output from the `transform` method.

        Returns
        -------
        pd.DataFrame with columns
            anomaly_interval: Intervals of the collective anomalies.
            anomaly_columns: Affected columns of the collective anomalies.
        """
        # The sparse format only uses integer positions, so we reset index and columns.
        y_dense = y_dense.reset_index(drop=True)
        y_dense.columns = range(y_dense.columns.size)

        anomaly_intervals = []
        unique_labels = np.unique(y_dense.values)
        for i in unique_labels[unique_labels > 0]:
            anomaly_mask = y_dense == i
            which_columns = anomaly_mask.any(axis=0)
            which_rows = anomaly_mask.any(axis=1)
            anomaly_columns = anomaly_mask.columns[which_columns].to_list()
            anomaly_start = anomaly_mask.index[which_rows][0]
            anomaly_end = anomaly_mask.index[which_rows][-1]
            anomaly_intervals.append((anomaly_start, anomaly_end, anomaly_columns))

        return SubsetCollectiveAnomalyDetector._format_sparse_output(
            anomaly_intervals, closed="both"
        )

    @staticmethod
    def _format_sparse_output(
        collective_anomalies: list[tuple[int, int, np.ndarray]],
        closed: str = "both",
    ) -> pd.DataFrame:
        """Format the sparse output of subset collective anomaly detectors.

        Can be reused by subclasses to format the output of the `_predict` method.

        Parameters
        ----------
        collective_anomalies : list
            List of tuples containing start and end indices of collective
            anomalies and a np.array of the affected components/columns.
        closed : str
            Whether the (start, end) tuple correspond to intervals that are closed
            on the left, right, both, or neither.

        Returns
        -------
        pd.DataFrame with columns
            anomaly_interval: Intervals of the collective anomalies.
            anomaly_columns: Affected columns of the collective anomalies.
        """
        anomaly_intervals = [(start, end) for start, end, _ in collective_anomalies]
        affected_components = [components for _, _, components in collective_anomalies]
        return pd.DataFrame(
            {
                "anomaly_interval": pd.IntervalIndex.from_tuples(
                    anomaly_intervals, closed=closed
                ),
                "anomaly_columns": affected_components,
            }
        )
