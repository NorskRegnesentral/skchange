"""Anomaly detection algorithms."""

from skchange.anomaly_detectors.anomalisers import StatThresholdAnomaliser
from skchange.anomaly_detectors.base import (
    CollectiveAnomalyDetector,
    SubsetCollectiveAnomalyDetector,
)
from skchange.anomaly_detectors.capa import CAPA
from skchange.anomaly_detectors.circular_binseg import CircularBinarySegmentation
from skchange.anomaly_detectors.mvcapa import MVCAPA

BASE_ANOMALY_DETECTORS = [
    CollectiveAnomalyDetector,
    SubsetCollectiveAnomalyDetector,
]
COLLECTIVE_ANOMALY_DETECTORS = [
    CAPA,
    CircularBinarySegmentation,
    MVCAPA,
    StatThresholdAnomaliser,
]
ANOMALY_DETECTORS = COLLECTIVE_ANOMALY_DETECTORS

__all__ = BASE_ANOMALY_DETECTORS + ANOMALY_DETECTORS
