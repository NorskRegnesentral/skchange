"""Base classes for changepoint detectors."""

import numpy as np
import pandas as pd

from skchange.base import BaseDetector


class ChangepointDetector(BaseDetector):
    """Base class for changepoint detectors.

    Changepoint detectors detect points in time where a change in the data occurs.
    Data between two changepoints is a segment where the data is considered to be
    homogeneous, i.e., of the same distribution. A changepoint is defined as the
    location of the last element of a segment.

    Output format of the predict method: See the dense_to_sparse method.
    Output format of the transform method: See the sparse_to_dense method.

    Subclasses should set the following tags for sktime compatibility:
    - task: "change_point_detection"
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
            The sparse output from a changepoint detector's predict method.
        index : array-like
            Indices that are to be annotated according to ``y_sparse``.

        Returns
        -------
        pd.Series
        """
        changepoints = y_sparse.to_list()
        n = len(index)
        changepoints = [-1] + changepoints + [n - 1]
        segment_labels = np.zeros(n)
        for i in range(len(changepoints) - 1):
            segment_labels[changepoints[i] + 1 : changepoints[i + 1] + 1] = i

        y_dense = pd.Series(
            segment_labels, index=index, name="segment_label", dtype="int64"
        )
        return y_dense

    @staticmethod
    def dense_to_sparse(y_dense: pd.Series) -> pd.Series:
        """Convert the dense output from the transform method to a sparse format.

        Parameters
        ----------
        y_dense : pd.Series
            The dense output from a changepoint detector's transform method.

        Returns
        -------
        pd.Series
        """
        y_dense = y_dense.reset_index(drop=True)
        # changepoint = end of segment, so the label diffs > 0 must be shiftet by -1.
        is_changepoint = np.roll(y_dense.diff().abs() > 0, -1)
        changepoints = y_dense.index[is_changepoint]
        y_sparse = pd.Series(changepoints, name="changepoint", dtype="int64")
        return y_sparse