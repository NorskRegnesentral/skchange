"""Anomaly detection algorithms."""

from ._anomalisers import StatThresholdAnomaliser
from ._capa import CAPA
from ._circular_binseg import CircularBinarySegmentation
from ._mvcapa import MVCAPA

COLLECTIVE_ANOMALY_DETECTORS = [
    CAPA,
    CircularBinarySegmentation,
    MVCAPA,
    StatThresholdAnomaliser,
]
ANOMALY_DETECTORS = COLLECTIVE_ANOMALY_DETECTORS

__all__ = ANOMALY_DETECTORS
