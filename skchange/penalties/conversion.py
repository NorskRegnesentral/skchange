"""Utilities for converting objects to penalties."""

import numbers
from typing import Union

from skchange.penalties.base import BasePenalty
from skchange.penalties.constant_penalties import ConstantPenalty


def as_constant_penalty(penalty: Union[BasePenalty, float]):
    """Convert an input object to a constant penalty.

    Parameters
    ----------
    penalty : BasePenalty, float, None
        Penalty to convert. If a float, a constant penalty with that value is created.
        If None, the default penalty is returned.

    Returns
    -------
    BasePenalty with ``penalty_type="constant"``
        Constant penalty object.
    """
    if isinstance(penalty, numbers.Number):
        out_penalty = ConstantPenalty(penalty)
    elif isinstance(penalty, BasePenalty) and penalty.penalty_type == "constant":
        out_penalty = penalty
    else:
        raise ValueError(
            "penalty must be a number or a BasePenalty object with"
            " 'constant' penalty_type."
        )
    return out_penalty
