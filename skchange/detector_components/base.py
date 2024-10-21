"""Base classes for detection scores."""

__author__ = ["Tveten"]

import inspect

import numpy as np
from numpy.typing import ArrayLike
from skbase.base import BaseObject

from skchange.utils.validation.parameters import check_jitted


class BaseDetectorComponent(BaseObject):
    """Base class template for detector components."""

    def __init__(self):
        self.jitted_precompute = None
        self.jitted_compute = None
        self._build_jitted_precompute()
        self._build_jitted_compute()
        self._check_build()

    def _build_jitted_precompute(self):
        raise NotImplementedError("abstract method")

    def _build_jitted_compute(self):
        raise NotImplementedError("abstract method")

    def _check_build(self):
        check_jitted(self.jitted_precompute)
        check_jitted(self.jitted_compute)
        self._check_jitted_precompute_input()
        self._check_jitted_compute_input()
        self._check_jitted_compute_output()
        # No checks are performed on the precompute output and inputs.
        # It is complicated to check this without being too restrictive.
        # It is left to the numba compiler to handle for now.

    def precompute(self, X: ArrayLike) -> np.ndarray | tuple:
        """Precompute parameters for the detector component."""
        return self.jitted_precompute(np.asarray(X))

    def compute(self, precomputed: np.ndarray | tuple, *args) -> np.ndarray:
        """Compute the detector component.

        Implemented by sub base classes for the different types of detector components,
        like costs, change scores, and anomaly scores. Note that the additional
        arguments beyond `precomputed` depend on the subclass. The signature of the
        `compute` is the template for the `jitted_compute` function to be implemented
        by the concrete detector component.

        """
        raise NotImplementedError("abstract method")

    def _check_jitted_precompute_input(self):
        """Check the input signature of jitted_precompute."""
        jitted_sig = inspect.signature(self.jitted_precompute)
        if len(jitted_sig.parameters) != 1:
            raise ValueError("jitted_precompute must take a single argument as input.")

        param = list(jitted_sig.parameters.values())[0]
        if param.annotation != np.ndarray:
            raise ValueError(
                "jitted_precompute must take a single np.ndarray as input."
            )

    def _check_jitted_compute_input(self):
        """Check the input signature of jitted_compute."""
        jitted_sig = inspect.signature(self.jitted_compute)
        method_sig = inspect.signature(self.compute)

        if len(jitted_sig.parameters) != len(method_sig.parameters):
            raise ValueError(
                "`jitted_compute` must have the same number of arguments as `compute`."
            )

        # Check that all but the first parameters to jitted_compute and method are the
        # same. The first parameter is supposed to be the precomputed output.
        jitted_params = list(jitted_sig.parameters.values())
        method_params = list(method_sig.parameters.values())
        for jitted_param, method_param in zip(jitted_params[1:], method_params[1:]):
            if jitted_param.annotation != method_param.annotation:
                raise ValueError(
                    f"The signature of `jitted_compute` {jitted_param.annotation} must"
                    + f"match the signature of `compute` {method_param.annotation}."
                )

    def _check_jitted_compute_output(self):
        """Check the output signature of jitted_compute."""
        if inspect.signature(self.jitted_compute).return_annotation != np.ndarray:
            raise ValueError("`jitted_compute` must return an np.ndarray.")


class BaseCost(BaseDetectorComponent):
    """Base class template for costs."""

    def compute(
        self, precomputed: np.ndarray | tuple, starts: np.ndarray, ends: np.ndarray
    ) -> np.ndarray:
        """Compute the cost for each segment.

        Parameters
        ----------
        precomputed : `tuple`
            Precomputed parameters from `jitted_precompute`.
        starts : `np.ndarray`
            Start indices of the segments (inclusive).
        ends : `np.ndarray`
            End indices of the segments (inclusive).

        Returns
        -------
        costs : `np.ndarray`
            Costs for each segment.
        """
        return self.jitted_compute(precomputed, starts, ends)


class BaseChangeScore(BaseDetectorComponent):
    """Base class template for change scores."""

    def compute(
        self,
        precomputed: np.ndarray | tuple,
        starts: np.ndarray,
        ends: np.ndarray,
        splits: np.ndarray,
    ) -> np.ndarray:
        """Compute the score for a changepoint within the segment."""
        return self.jitted_compute(precomputed, starts, ends, splits)


class BaseAnomalyScore(BaseDetectorComponent):
    """Base class template for anomaly scores."""

    def compute(
        self,
        precomputed: np.ndarray | tuple,
        starts: np.ndarray,
        ends: np.ndarray,
        anomaly_starts: np.ndarray,
        anomaly_ends: np.ndarray,
    ) -> np.ndarray:
        """Compute the score for an anomaly within the segment."""
        return self.jitted_compute(
            precomputed, starts, ends, anomaly_starts, anomaly_ends
        )


class BaseSaving(BaseDetectorComponent):
    """Base class template for savings.

    Savings are anomaly scores where the baseline parameters are assumed known.
    """

    def compute(
        self, precomputed: np.ndarray | tuple, starts: np.ndarray, ends: np.ndarray
    ) -> np.ndarray:
        """Compute the saving for each segment."""
        return self.jitted_compute(precomputed, starts, ends)
