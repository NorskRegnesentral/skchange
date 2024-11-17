"""Utility functions for anomaly scores."""

from typing import Union

from skchange.anomaly_scores import BaseSaving, Saving
from skchange.costs.base import BaseCost


def to_saving(evaluator: Union[BaseCost, BaseSaving]) -> BaseSaving:
    """Convert a cost function to a saving.

    Parameters
    ----------
    evaluator : BaseCost or BaseSaving
        The evalutor to convert to a saving. If a cost, it must be a cost with a fixed
        parameter. If a saving is provided, it is returned as is.

    Returns
    -------
    saving : BaseSaving
        The saving based on the cost function.
    """
    if isinstance(evaluator, BaseCost):
        saving = Saving(cost=evaluator)
    elif isinstance(evaluator, BaseSaving):
        saving = evaluator
    else:
        raise ValueError(
            f"evaluator must be an instance of BaseSaving or BaseCost. "
            f"Got {type(saving)}."
        )
    return saving
