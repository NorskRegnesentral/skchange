"""Anomaly scores for interval evaluation."""

from skchange.anomaly_scores.base import BaseLocalAnomalyScore, BaseSaving
from skchange.anomaly_scores.from_cost import (
    LocalAnomalyScore,
    Saving,
    to_local_anomaly_score,
    to_saving,
)
from skchange.anomaly_scores.l2_saving import L2Saving

BASE_LOCAL_ANOMALY_SCORES = [
    BaseLocalAnomalyScore,
]
LOCAL_ANOMALY_SCORES = [
    LocalAnomalyScore,
]
BASE_SAVINGS = [
    BaseSaving,
]
SAVINGS = [
    Saving,
    L2Saving,
]
BASE_ANOMALY_SCORES = BASE_SAVINGS + BASE_LOCAL_ANOMALY_SCORES
ANOMALY_SCORES = SAVINGS + LOCAL_ANOMALY_SCORES

__all__ = (
    BASE_ANOMALY_SCORES
    + ANOMALY_SCORES
    + [
        to_local_anomaly_score,
        to_saving,
    ]
)
