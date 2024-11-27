"""Anomaly detection algorithms."""

from skchange.anomaly_detectors.anomalisers import StatThresholdAnomaliser
from skchange.anomaly_detectors.base import (
    CollectiveAnomalyDetector,
    PointAnomalyDetector,
    SubsetCollectiveAnomalyDetector,
)
from skchange.anomaly_detectors.capa import CAPA
from skchange.anomaly_detectors.circular_binseg import CircularBinarySegmentation
from skchange.anomaly_detectors.mvcapa import Mvcapa

BASE_ANOMALY_DETECTORS = [
    CollectiveAnomalyDetector,
    PointAnomalyDetector,
    SubsetCollectiveAnomalyDetector,
]
COLLECTIVE_ANOMALY_DETECTORS = [
    CAPA,
    CircularBinarySegmentation,
    Mvcapa,
    StatThresholdAnomaliser,
]
POINT_ANOMALY_DETECTORS = []
ANOMALY_DETECTORS = COLLECTIVE_ANOMALY_DETECTORS + POINT_ANOMALY_DETECTORS

__all__ = BASE_ANOMALY_DETECTORS + ANOMALY_DETECTORS
