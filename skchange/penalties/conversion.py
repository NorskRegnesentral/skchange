"""Utilities for converting objects to penalties."""

import numbers
from typing import Union

import numpy as np

from skchange.penalties.base import BasePenalty
from skchange.penalties.constant_penalties import BICPenalty, ConstantPenalty
from skchange.penalties.linear_penalties import LinearPenalty
from skchange.penalties.nonlinear_penalties import NonlinearPenalty


def as_penalty(
    x: Union[BasePenalty, np.ndarray, tuple[float, float], float, None],
    default: BasePenalty = None,
    require_penalty_type: str = None,
):
    """Convert an input object to a constant penalty.

    Parameters
    ----------
    x : BasePenalty, np.ndarray, tuple[float, float], float, None
        Object to convert to a penalty.
    default : BasePenalty, optional, default=BICPenalty
        Default penalty to return if penalty is None.
    require_penalty_type: str, optional, default=None
        Required penalty type. If not None, the penalty must be of this type.

    Returns
    -------
    penalty : BasePenalty
        Penalty object.
    """
    default_penalty = BICPenalty() if default is None else default

    if x is None:
        penalty = default_penalty
    elif isinstance(x, numbers.Number) or (isinstance(x, np.ndarray) and x.size == 1):
        penalty = ConstantPenalty(x)
    elif isinstance(x, tuple) and len(x) == 2:
        penalty = LinearPenalty(*x)
    elif isinstance(x, np.ndarray):
        x_diff = np.diff(x)
        if np.all(x_diff == x_diff[0]):
            slope = x_diff[0]
            intercept = x[0] - slope
            penalty = LinearPenalty(intercept, slope)
        else:
            penalty = NonlinearPenalty(x)
    elif isinstance(x, BasePenalty):
        penalty = x
    else:
        raise ValueError(
            f"Cannot convert {type(x)} to a penalty."
            " Expected a number, numpy array, 2-tuple, or BasePenalty object."
            f" Got {x}."
        )

    if require_penalty_type is not None:
        valid_penalty_types = ["constant", "linear", "nonlinear"]
        if require_penalty_type not in valid_penalty_types:
            raise ValueError(
                f"Invalid penalty type: {require_penalty_type}."
                f" Expected one of {valid_penalty_types}."
            )

        if penalty.penalty_type != require_penalty_type:
            raise ValueError(
                f"{x} cannot be converted to a {require_penalty_type} penalty."
            )

    return penalty
