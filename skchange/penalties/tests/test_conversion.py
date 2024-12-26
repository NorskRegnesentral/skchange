"""Tests for penalty conversion utilities."""

import pytest

from skchange.penalties.base import BasePenalty
from skchange.penalties.constant_penalties import ConstantPenalty
from skchange.penalties.conversion import as_constant_penalty


class MockPenalty(BasePenalty):
    def __init__(self, penalty_type):
        self.penalty_type = penalty_type


def test_as_constant_penalty_with_float():
    penalty = 5.0
    result = as_constant_penalty(penalty)
    assert isinstance(result, ConstantPenalty)
    assert result.values[0] == penalty


def test_as_constant_penalty_with_constant_penalty():
    penalty = ConstantPenalty(5.0)
    result = as_constant_penalty(penalty)
    assert result is penalty


def test_as_constant_penalty_with_invalid_penalty_type():
    penalty = MockPenalty("non-constant")
    with pytest.raises(ValueError):
        as_constant_penalty(penalty)


def test_as_constant_penalty_with_invalid_type():
    penalty = "invalid"
    with pytest.raises(ValueError):
        as_constant_penalty(penalty)
