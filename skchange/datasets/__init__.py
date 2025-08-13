"""Datasets and dataset generators for skchange."""

from ._data_loaders import load_hvac_system_data
from ._generate import (
    generate_piecewise_data,
)
from ._generate_linear_trend import generate_continuous_piecewise_linear_signal
from ._generate_normal import (
    generate_alternating_data,
    generate_anomalous_data,
    generate_changing_data,
    generate_piecewise_normal_data,
)
from ._generate_regression import generate_piecewise_regression_data

__all__ = [
    "generate_anomalous_data",
    "generate_alternating_data",
    "generate_changing_data",
    "generate_continuous_piecewise_linear_signal",
    "load_hvac_system_data",
    "generate_piecewise_data",
    "generate_piecewise_normal_data",
    "generate_piecewise_regression_data",
]
