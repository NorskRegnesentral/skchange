"""Base classes for detection scores."""

__author__ = ["Tveten"]

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from skbase.base import BaseObject

from skchange.utils.numba.njit import njit
from skchange.utils.validation.parameters import check_jitted


def convert_2d_array(self, X: ArrayLike) -> np.ndarray:
    """Convert input to a 2D array."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim != 2:
        raise ValueError("Input must be a 1D or 2D array.")
    return X


class BaseEvaluator(BaseObject):
    """Base class template for evaluation functions on lists of datasets."""

    def _evaluate(self, X) -> float:
        """Evaluate on a dataset."""
        raise NotImplementedError("abstract method")

    def evaluate(self, X) -> float:
        """Evaluate on a dataset."""
        X = self._check_input(X)

        return self._evaluate(X)

    def _check_input(self, X):
        return X


class NumbaEvaluatorMixin:
    """Mixin for evaluation functions based on numba."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_evaluate()
        self._check_build()

    def _check_build(self):
        check_jitted(self._njit_evaluate)

    def _build_evaluate(self):
        """Build the evaluate function."""
        # @njit
        # def njit_evaluate(X: np.ndarray | list[np.ndarray]) -> float:
        #     <code>
        #     return value
        #
        # self._njit_evaluate = njit_evaluate

        raise NotImplementedError("abstract method")

    @staticmethod
    def _njit_evaluate(X: np.ndarray | list[np.ndarray]) -> float:
        """Evaluate on a list of datasets."""
        raise NotImplementedError("abstract method")

    def _evaluate(self, X: np.ndarray | list[np.ndarray]) -> float:
        if isinstance(ArrayLike):
            X = convert_2d_array(X)
        return self._njit_evaluate(X)

    def _check_input(
        self, X: ArrayLike | list[ArrayLike]
    ) -> np.ndarray | list[np.ndarray]:
        if isinstance(X, ArrayLike):
            X = convert_2d_array(X)

        if not isinstance(X, (ArrayLike, list)):
            raise ValueError("Input must be an array-like or list of array-like.")
        if isinstance(X, list) and not all(isinstance(x, ArrayLike) for x in X):
            raise ValueError("All elements in the list must be array-like.")
        return X


class BaseCost(BaseEvaluator):
    """Base class template for cost functions."""

    def __init__(self, param=None):
        super().__init__()
        self.param = param

    @staticmethod
    def min_size(X) -> int:
        """Minimum size of dataset to evaluate.

        May depend on the input dataset X.
        """
        return 1


class L2Cost(NumbaEvaluatorMixin, BaseCost):
    """L2 cost function."""

    def __init__(self, param: Optional[float | np.ndarray] = None):
        super().__init__(param)

    def _build_evaluate(self):
        param = self.param
        min_size = njit(self.min_size)

        @njit
        def njit_evaluate(X: np.ndarray) -> float:
            n = X.shape[0]
            if n < min_size(X):
                return np.inf

            mean = np.sum(X, axis=0) / n if param is None else param
            univar_costs = np.sum((X - mean) ** 2, axis=0)
            cost = np.sum(univar_costs)
            return cost

        self._njit_evaluate = njit_evaluate

    def _check_input(self, X):
        if not isinstance(X, ArrayLike):
            raise ValueError("Input must be an array-like.")
