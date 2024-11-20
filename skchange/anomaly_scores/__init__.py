"""Anomaly scores for interval evaluation."""

from skchange.anomaly_scores.base import BaseSaving
from skchange.anomaly_scores.from_cost import Saving, to_saving

BASE_ANOMALY_SCORES = [
    BaseSaving,
]
ANOMALY_SCORES = [
    Saving,
]

__all__ = (
    BASE_ANOMALY_SCORES
    + ANOMALY_SCORES
    + [
        to_saving,
    ]
)
