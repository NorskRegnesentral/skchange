"""Datasets and dataset generators for skchange."""

from ._data_loaders import load_hvac_system_data
from ._generate import (
    add_linspace_outliers,
    generate_alternating_data,
    generate_anomalous_data,
    generate_changing_data,
)

__all__ = [
    "add_linspace_outliers",
    "generate_anomalous_data",
    "generate_alternating_data",
    "generate_changing_data",
    "load_hvac_system_data",
]
