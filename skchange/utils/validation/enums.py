"""Representation of values with restricted domains."""

from enum import Enum


class EvaluationType(Enum):
    """Different types of evaluation for the change point detection algorithms."""

    UNIVARIATE = 1
    MULTIVARIATE = 2
