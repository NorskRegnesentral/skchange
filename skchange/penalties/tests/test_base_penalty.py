"""Test the base penalty class."""

import pytest

from skchange.penalties.base import BasePenalty


def test_base_values_not_implemented():
    penalty = BasePenalty()
    with pytest.raises(NotImplementedError):
        penalty._base_values


def test_not_fitted_error():
    penalty = BasePenalty()
    with pytest.raises(ValueError):
        penalty.values
