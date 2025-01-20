"""Penalties and penalty functions for change and anomaly detection."""

from skchange.penalties.base import BasePenalty
from skchange.penalties.composition import MinimumPenalty
from skchange.penalties.constant_penalties import (
    BICPenalty,
    ChiSquarePenalty,
    ConstantPenalty,
)
from skchange.penalties.conversion import as_penalty
from skchange.penalties.linear_penalties import LinearChiSquarePenalty, LinearPenalty
from skchange.penalties.nonlinear_penalties import (
    NonlinearChiSquarePenalty,
    NonlinearPenalty,
)

PENALTIES = [
    ConstantPenalty,
    BICPenalty,
    ChiSquarePenalty,
    LinearChiSquarePenalty,
    LinearPenalty,
    NonlinearChiSquarePenalty,
    NonlinearPenalty,
    MinimumPenalty,
]

__all__ = [
    "BasePenalty",
    "BICPenalty",
    "ChiSquarePenalty",
    "ConstantPenalty",
    "LinearChiSquarePenalty",
    "LinearPenalty",
    "MinimumPenalty",
    "NonlinearChiSquarePenalty",
    "NonlinearPenalty",
    "as_penalty",
]
