"""Penalties and penalty functions for change and anomaly detection."""

from skchange.penalties.base import BasePenalty
from skchange.penalties.composition import MinimumPenalty
from skchange.penalties.constant_penalties import (
    BICPenalty,
    ChiSquarePenalty,
    ConstantPenalty,
)
from skchange.penalties.linear_penalties import SparseChiSquarePenalty
from skchange.penalties.nonlinear_penalties import IntermediateChiSquarePenalty

PENALTIES = [
    ConstantPenalty,
    BICPenalty,
    ChiSquarePenalty,
    SparseChiSquarePenalty,
    IntermediateChiSquarePenalty,
]

__all__ = [
    "BasePenalty",
    "BICPenalty",
    "ChiSquarePenalty",
    "ConstantPenalty",
    "SparseChiSquarePenalty",
]
