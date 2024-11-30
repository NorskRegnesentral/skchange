"""Datasets and dataset generators for skchange."""

from skchange.datasets._data_loaders import load_air_handling_unit
from skchange.datasets.generate import (
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
    "load_air_handling_unit",
]
