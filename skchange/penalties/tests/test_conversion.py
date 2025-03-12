"""Tests for penalty conversion utilities."""

import numpy as np
import pytest

from skchange.penalties import (
    BICPenalty,
    ConstantPenalty,
    LinearPenalty,
    NonlinearPenalty,
    as_penalty,
)
from skchange.penalties.base import BasePenalty


class MockPenalty(BasePenalty):
    def __init__(self, penalty_type):
        self.penalty_type = penalty_type


def test_as_penalty_with_none():
    result = as_penalty(None)
    assert isinstance(result, BICPenalty)


def test_as_penalty_with_float():
    penalty = 5.0
    result = as_penalty(penalty)
    assert isinstance(result, ConstantPenalty)


def test_as_penalty_with_ndarray_single_value():
    penalty = np.array([5.0])
    result = as_penalty(penalty)
    assert isinstance(result, ConstantPenalty)


def test_as_penalty_with_tuple():
    penalty = (2.0, 3.0)
    result = as_penalty(penalty)
    assert isinstance(result, LinearPenalty)
    assert result.intercept == penalty[0]
    assert result.slope == penalty[1]


def test_as_penalty_with_linear_ndarray():
    penalty = np.array([1.0, 2.0, 3.0])
    result = as_penalty(penalty)
    assert isinstance(result, LinearPenalty)
    assert result.slope == 1.0
    assert result.intercept == 0.0


def test_as_penalty_with_nonlinear_ndarray():
    penalty = np.array([1.0, 2.0, 4.0])
    result = as_penalty(penalty)
    assert isinstance(result, NonlinearPenalty)


def test_as_penalty_with_base_penalty():
    penalty = MockPenalty("constant")
    result = as_penalty(penalty)
    assert result is penalty


def test_as_penalty_with_invalid_type():
    penalty = "invalid"
    with pytest.raises(ValueError):
        as_penalty(penalty)


def test_as_penalty_with_invalid_penalty_type():
    penalty = MockPenalty("non-constant")
    with pytest.raises(ValueError):
        as_penalty(penalty, require_penalty_type="constant")


def test_as_penalty_with_invalid_required_penalty_type():
    penalty = 5.0
    with pytest.raises(ValueError):
        as_penalty(penalty, require_penalty_type="invalid")
