"""Utility functions for change scores."""

from typing import Union

from skchange.change_scores import BaseChangeScore, ChangeScore
from skchange.costs.base import BaseCost


def to_change_score(evaluator: Union[BaseCost, BaseChangeScore]) -> BaseChangeScore:
    """Convert a cost function to a change score.

    Parameters
    ----------
    evaluator : BaseCost or BaseChangeScore
        The evalutor to convert to a change score. If a change score is provided, it is
        returned as is.

    Returns
    -------
    change_score : BaseChangeScore
        The change score based on the cost function.
    """
    if isinstance(evaluator, BaseCost):
        change_score = ChangeScore(cost=evaluator)
    elif isinstance(evaluator, BaseChangeScore):
        change_score = evaluator
    else:
        raise ValueError(
            f"evaluator must be an instance of BaseChangeScore or BaseCost. "
            f"Got {type(evaluator)}."
        )
    return change_score
