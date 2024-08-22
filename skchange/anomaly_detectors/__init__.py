"""Anomaly detection algorithms."""

from skchange.anomaly_detectors.anomalisers import StatThresholdAnomaliser
from skchange.anomaly_detectors.base import (
    CollectiveAnomalyDetector,
    PointAnomalyDetector,
)
from skchange.anomaly_detectors.capa import Capa

BASE_ANOMALY_DETECTORS = [CollectiveAnomalyDetector, PointAnomalyDetector]
COLLECTIVE_ANOMALY_DETECTORS = [
    # Capa,
    # CircularBinarySegmentation,
    # MoscoreAnomaly,
    # Mvcapa,
    StatThresholdAnomaliser,
]
POINT_ANOMALY_DETECTORS = []
ANOMALY_DETECTORS = COLLECTIVE_ANOMALY_DETECTORS + POINT_ANOMALY_DETECTORS

__all__ = BASE_ANOMALY_DETECTORS + ANOMALY_DETECTORS
