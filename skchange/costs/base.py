"""Cost functions as interval evaluators."""

import numpy as np

from ..base import BaseIntervalScorer


class BaseCost(BaseIntervalScorer):
    """Base class template for cost functions.

    This is a common base class for cost functions. It is used to evaluate a cost
    function on a set of intervals

    If the cost supports fixed parameters, that is indicated by the
    `supports_fixed_params` class attribute. By default, this is set to `False`.
    If the cost supports fixed parameters, the `param` attribute can be set
    in the constructor, and the fixed `param` paramaters will then used when
    evaluating the cost. The type of `param` is specific to each concrete cost.

    Parameters
    ----------
    param : None, optional (default=None)
        If ``None``, the cost is evaluated with parameters
        that minimize the cost.  If ```param`` is not ``None``,
        the cost is evaluated at that fixed parameter.
        The parameter type is specific to each concrete cost.
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": "Tveten",
    }

    supports_fixed_params = False
    expected_cut_entries = 2

    def __init__(self, param=None):
        self.param = param
        super().__init__()
        if self.param is not None and not self.supports_fixed_params:
            raise ValueError("This cost does not support fixed parameters.")

    def _check_param(self, param, X):
        """Check the parameter with respect to the input data.

        This method should be called in `_fit` of subclasses.
        """
        if param is None:
            return None
        return self._check_fixed_param(param, X)

    def _check_fixed_param(self, param, X):
        """Check the fixed parameter with respect to the input data.

        This method defaults to no checking, but it should be overwritten in subclasses
        to make sure `param` is valid relative to the input data `X`.
        """
        return param

    def evaluate_segmentation(self, segmentation: np.ndarray) -> float:
        """Evaluate the cost of a segmentation.

        # IDEA: Implement 'CROPS' algoritghm: https://arxiv.org/pdf/1412.3617

        Parameters
        ----------
        segmentation : np.ndarray
            A 1D array with the indices of the change points in the input data.
            Each change point signifies the first index of a new segment.

        Returns
        -------
        cost : float
            The cost of the segmentation.
        """
        if segmentation.ndim != 1:
            raise ValueError("The segmentation must be a 1D array.")
        # Prepend 0 and append the length of the data to the segmentation:
        # This is done to ensure that the first and last segments are included.
        # The segmentation is assumed to be sorted in increasing order.
        if np.any(np.diff(segmentation) <= 0):
            raise ValueError(
                "The segmentation must be a 1D array with strictly increasing entries."
            )
        if len(segmentation) == 0:
            segmentation = np.array([0, self._X.shape[0]])
        elif segmentation[0] != 0 and segmentation[-1] != self._X.shape[0]:
            segmentation = np.concatenate(
                (np.array([0]), segmentation, np.array([self._X.shape[0]]))
            )
        cuts = np.vstack((segmentation[:-1], segmentation[1:])).T
        # Currently supports only "multivariate" evaluation type, returns a single
        # cost value.
        return np.sum(self.evaluate(cuts))

    def _evaluate(self, cuts: np.ndarray) -> np.ndarray:
        """Evaluate the cost on a set of intervals.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array with two columns of integer location-based intervals to evaluate.
            The subsets ``X[cuts[i, 0]:cuts[i, 1]]`` for
            ``i = 0, ..., len(cuts)`` are evaluated.

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of
            columns is 1 if the cost is inherently multivariate. The number of columns
            is equal to the number of columns in the input data if the cost is
            univariate. In this case, each column represents the univariate cost for
            the corresponding input data column.
        """
        starts, ends = cuts[:, 0], cuts[:, 1]
        if self.param is None:
            costs = self._evaluate_optim_param(starts, ends)
        else:
            costs = self._evaluate_fixed_param(starts, ends)

        return costs

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the optimal parameter.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of
            columns is 1 if the cost is inherently multivariate. The number of columns
            is equal to the number of columns in the input data if the cost is
            univariate. In this case, each column represents the univariate cost for
            the corresponding input data column.
        """
        raise NotImplementedError("abstract method")

    def _evaluate_fixed_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the fixed parameter.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of
            columns is 1 if the cost is inherently multivariate. The number of columns
            is equal to the number of columns in the input data if the cost is
            univariate. In this case, each column represents the univariate cost for
            the corresponding input data column.
        """
        raise NotImplementedError("abstract method")
