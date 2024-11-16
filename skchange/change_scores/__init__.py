"""Change scores as interval evaluators."""

from skchange.change_scores.base import BaseChangeScore
from skchange.change_scores.cost_based import ChangeScore

BASE_CHANGE_SCORES = [
    BaseChangeScore,
]
CHANGE_SCORES = [
    ChangeScore,
]

__all__ = BASE_CHANGE_SCORES + CHANGE_SCORES
