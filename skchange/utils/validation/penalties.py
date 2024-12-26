"""Validation functions for penalties."""

from numbers import Number
from typing import Union

from skchange.penalties import BasePenalty


def check_constant_penalty(
    penalty: Union[BasePenalty, float, None], caller=None, allow_none=False
):
    """Check if the penalty is coercible to a constant penalty.

    Parameters
    ----------
    penalty : BasePenalty, float, None
        Penalty to check.
    caller : object, optional, default=None
        Object calling the function. If not None, the error message will be more
        specific.
    allow_none : bool, optional, default=False
        Whether to allow None penalties.

    Raises
    ------
    ValueError
        If the penalty is not coercible to a constant penalty.
    """
    if isinstance(penalty, BasePenalty) and penalty.penalty_type != "constant":
        if caller is None:
            raise ValueError(
                "The penalty must be a constant penalty."
                f" Got type {penalty.penalty_type}."
            )
        else:
            raise ValueError(
                f"{caller.__class__} only supports constant penalties."
                f" Got type {penalty.penalty_type}."
            )

    if penalty is not None and not isinstance(penalty, (Number, BasePenalty)):
        raise ValueError(
            f"penalty must be None, a number or a BasePenalty object. Got {penalty}."
        )

    if not allow_none and penalty is None:
        raise ValueError("penalty cannot be None.")
