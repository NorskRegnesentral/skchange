"""Utilities for converting objects to penalties."""

import numbers
from typing import Union

from skchange.penalties.base import BasePenalty
from skchange.penalties.constant_penalties import BICPenalty, ConstantPenalty


def as_constant_penalty(
    penalty: Union[BasePenalty, float, None],
    default: BasePenalty = None,
    caller=None,
):
    """Convert an input object to a constant penalty.

    Parameters
    ----------
    penalty : BasePenalty, float, None
        Penalty to convert. If a float, a constant penalty with that value is created.
        If None, the default penalty is returned.
    default : BasePenalty, default=BICPenalty()
        Default penalty to return if penalty is None.
    caller : object, default=None
        Object calling the function. Used for more informative error messages.

    Returns
    -------
    BasePenalty with ``penalty_type="constant"``
        Constant penalty object.
    """
    default = BICPenalty() if default is None else default

    if penalty is None:
        out_penalty = default
    if isinstance(penalty, numbers.Number):
        out_penalty = ConstantPenalty(penalty)
    elif isinstance(penalty, BasePenalty):
        out_penalty = penalty
    else:
        raise ValueError("penalty must be None, a number or a BasePenalty object.")

    if out_penalty.penalty_type != "constant":
        if caller is None:
            raise ValueError(
                "The penalty must be a constant penalty."
                f" Got type {out_penalty.penalty_type}."
            )
        else:
            raise ValueError(
                f"{caller.__class__} only supports constant penalties."
                f" Got type {out_penalty.penalty_type}."
            )

    return out_penalty
