"""Base classes for detection scores."""

__author__ = ["Tveten"]

import inspect
from typing import Callable

import numpy as np
from skbase.base import BaseObject

from skchange.numba_subset_evaluators.utils import identity
from skchange.utils.numba.njit import njit
from skchange.utils.validation.parameters import check_jitted


def build_default_evaluate_vectorized(subset: Callable, evaluate: Callable):
    """Build a default evaluate_vectorized function."""
    check_jitted(subset)
    check_jitted(evaluate)

    @njit
    def default_evaluate_vectorized(
        X: np.ndarray,
        subsetter: np.ndarray,
    ) -> np.ndarray:
        n_evals = subsetter.shape[0]
        values = np.zeros(n_evals, dtype=np.float64)
        for i in range(n_evals):
            values[i] = evaluate(subset(X, subsetter[i]))
        return values

    return default_evaluate_vectorized


class NumbaSubsetEvaluator(BaseObject):
    """Base class template for detector components."""

    def __init__(self):
        self._subset = None
        self._evaluate = None
        self._precompute = None
        self._evaluate_vectorized = None

        self._build_subset()
        self._build_evaluate()
        self._build_precompute()
        self._build_evaluate_vectorized()
        self._check_build()
        super().__init__()

    def _build_subset(self):
        # def _subset(X: np.ndarray, subsetter: np.ndarray) -> np.ndarray:
        #     <code>
        #     return value
        #
        # self._subset = _subset
        raise NotImplementedError("abstract method")

    def _subset(
        self, X: np.ndarray, subsetter: np.ndarray
    ) -> np.ndarray | list[np.ndarray]:
        raise NotImplementedError("abstract method")

    def _build_evaluate(self):
        # def evaluate(X_subsets: np.ndarray | list[np.ndarray]) -> float:
        #     <code>
        #     return value
        #
        # self._evaluate = evaluate
        raise NotImplementedError("abstract method")

    @staticmethod
    def _evaluate(X_subsets: np.ndarray | list[np.ndarray]) -> float:
        raise NotImplementedError("abstract method")

    def evaluate(self, X_subsets: np.ndarray | list[np.ndarray]) -> float:
        """Evaluate on subsets."""
        return self._evaluate(X_subsets)

    def _build_precompute(self):
        self._precompute = identity

    def precompute(self, X: np.ndarray) -> np.ndarray | list[np.ndarray]:
        """Precompute parameters for `evaluate_vectorized`."""
        return self._precompute(X)

    def _build_evaluate_vectorized(self):
        self._evaluate_vectorized = build_default_evaluate_vectorized(
            self._subset, self._evaluate
        )

    def evaluate_vectorized(
        self,
        precomputed: np.ndarray | list[np.ndarray],
        subsetter: np.ndarray,
    ) -> np.ndarray:
        """Evaluate over several segments according to splits."""
        return self._evaluate_vectorized(precomputed, subsetter)

    def _check_build(self):
        check_jitted(self._subset)
        check_jitted(self._evaluate)
        check_jitted(self._precompute)
        check_jitted(self._evaluate_vectorized)
        self._check_subset_input()
        self._check_subset_output()
        self._check_evaluate_input()
        self._check_evaluate_output()
        self._check_precompute_input()
        self._check_evaluate_vectorized_input()
        self._check_evaluate_vectorized_output()
        # No checks are performed on the precompute output and inputs.
        # It is complicated to check this without being too restrictive.
        # It is left to the numba compiler to handle for now.

    def _check_subset_input(self):
        pass

    def _check_subset_output(self):
        pass

    def _check_evaluate_input(self):
        pass

    def _check_evaluate_output(self):
        pass

    def _check_precompute_input(self):
        """Check the input signature of _precompute."""
        sig = inspect.signature(self._precompute)
        if len(sig.parameters) != 1:
            raise ValueError("_precompute must take a single argument as input.")

        param = list(sig.parameters.values())[0]
        if param.annotation != np.ndarray:
            raise ValueError("_precompute must take a single np.ndarray as input.")

    def _check_evaluate_vectorized_input(self):
        """Check the input signature of _evaluate_vectorized."""
        private_sig = inspect.signature(self._evaluate_vectorized)
        public_sig = inspect.signature(self.evaluate_vectorized)

        if len(private_sig.parameters) != len(public_sig.parameters):
            raise ValueError(
                "`_evaluate_vectorized` must have the same number of arguments as"
                + " `evaluate_vectorized`."
            )

        # Check that all but the first parameters to _evaluate_vectorized and method are
        # the same. The first parameter is supposed to be the precomputed output.
        private_params = list(private_sig.parameters.values())
        public_params = list(public_sig.parameters.values())
        for private_param, public_param in zip(private_params[1:], public_params[1:]):
            private_annotation = private_param.annotation
            public_annotation = public_param.annotation
            if private_annotation != public_annotation:
                raise ValueError(
                    f"The signature of `_evaluate_vectorized` {private_annotation} must"
                    + " match the signature of `evaluate_vectorized"
                    + f" {public_annotation}."
                )

    def _check_evaluate_vectorized_output(self):
        """Check the output signature of _evaluate_vectorized."""
        if inspect.signature(self._evaluate_vectorized).return_annotation != np.ndarray:
            raise ValueError("`_evaluate_vectorized` must return an np.ndarray.")
