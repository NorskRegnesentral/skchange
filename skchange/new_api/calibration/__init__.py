"""Calibration utilities for controlling false alarm rates in change detection."""

from skchange.new_api.calibration._calibrate import calibrate_penalty
from skchange.new_api.calibration._calibrated_detector import CalibratedDetector
from skchange.new_api.calibration._null_models import (
    BaseNullModel,
    BlockBootstrapNullModel,
    GaussianNullModel,
    MCNullModel,
    PermutationNullModel,
)

__all__ = [
    "BaseNullModel",
    "BlockBootstrapNullModel",
    "CalibratedDetector",
    "GaussianNullModel",
    "MCNullModel",
    "PermutationNullModel",
    "calibrate_penalty",
]
