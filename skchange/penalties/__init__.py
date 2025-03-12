"""Penalties and penalty functions for change and anomaly detection."""

from ._compose import MinimumPenalty
from ._constant_penalties import (
    BICPenalty,
    ChiSquarePenalty,
    ConstantPenalty,
)
from ._conversion import as_penalty
from ._linear_penalties import LinearChiSquarePenalty, LinearPenalty
from ._nonlinear_penalties import (
    NonlinearChiSquarePenalty,
    NonlinearPenalty,
)

PENALTIES = [
    BICPenalty,
    ChiSquarePenalty,
    ConstantPenalty,
    LinearChiSquarePenalty,
    LinearPenalty,
    NonlinearChiSquarePenalty,
    NonlinearPenalty,
    MinimumPenalty,
]

__all__ = PENALTIES + [
    "as_penalty",
]
