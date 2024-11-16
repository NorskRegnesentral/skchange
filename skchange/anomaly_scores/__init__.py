"""Anomaly scores for interval evaluation."""

from skchange.anomaly_scores.base import BaseSaving
from skchange.anomaly_scores.cost_based import Saving

BASE_ANOMALY_SCORES = [
    BaseSaving,
]
ANOMALY_SCORES = [
    Saving,
]

__all__ = BASE_ANOMALY_SCORES + ANOMALY_SCORES
