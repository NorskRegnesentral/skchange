"""Representation of values with restricted domains."""

from enum import Enum


class EvaluationType(Enum):
    """Different types of evaluation for the change point detection algorithms.

    * Univariate: Each variable is evaluated independently. The output of scorers have
      the same number of columns as the input data.
    * Multivariate: All variables are evaluated together. The output of scorers only
      has one column.
    * Conditional: Some variables are used as a response and others as covariates in a
      regression model. The output of scorers have the same number of columns as the
      number of response variables.
    """

    UNIVARIATE = 1
    MULTIVARIATE = 2
    CONDITIONAL = 3
