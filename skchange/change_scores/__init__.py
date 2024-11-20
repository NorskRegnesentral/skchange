"""Change scores as interval evaluators."""

from skchange.change_scores.base import BaseChangeScore
from skchange.change_scores.from_cost import ChangeScore, cost_to_change_score

BASE_CHANGE_SCORES = [
    BaseChangeScore,
]
CHANGE_SCORES = [
    ChangeScore,
]

__all__ = (
    BASE_CHANGE_SCORES
    + CHANGE_SCORES
    + [
        cost_to_change_score,
    ]
)
