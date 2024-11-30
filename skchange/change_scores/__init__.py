"""Change scores as interval evaluators."""

from skchange.change_scores.base import BaseChangeScore
from skchange.change_scores.cusum import CUSUM
from skchange.change_scores.from_cost import ChangeScore, to_change_score
from skchange.change_scores.gaussian_cov_score import GaussianCovScore

BASE_CHANGE_SCORES = [
    BaseChangeScore,
]
CHANGE_SCORES = [
    ChangeScore,
    GaussianCovScore,
    CUSUM,
]

__all__ = (
    BASE_CHANGE_SCORES
    + CHANGE_SCORES
    + [
        to_change_score,
    ]
)
