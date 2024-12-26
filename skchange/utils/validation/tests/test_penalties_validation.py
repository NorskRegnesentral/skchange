"""Tests for penalty validation utilities."""

import pytest

from skchange.penalties.base import BasePenalty
from skchange.penalties.constant_penalties import ConstantPenalty
from skchange.utils.validation.penalties import check_constant_penalty


class MockPenalty(BasePenalty):
    penalty_type = "mock"


def test_check_constant_penalty_with_none():
    with pytest.raises(ValueError, match="penalty cannot be None."):
        check_constant_penalty(None)


def test_check_constant_penalty_with_none_allowed():
    check_constant_penalty(None, allow_none=True)


def test_check_constant_penalty_with_float():
    check_constant_penalty(10.0)


def test_check_constant_penalty_with_base_penalty():
    penalty = ConstantPenalty(5.0)
    check_constant_penalty(penalty)


def test_check_constant_penalty_with_invalid_type():
    with pytest.raises(ValueError):
        check_constant_penalty("invalid")


def test_check_constant_penalty_with_non_constant_penalty():
    penalty = MockPenalty()
    with pytest.raises(ValueError, match="constant penalty"):
        check_constant_penalty(penalty)


def test_check_constant_penalty_with_non_constant_penalty_and_caller():
    penalty = MockPenalty()
    caller = object()
    with pytest.raises(
        ValueError,
        match=f"{caller.__class__}",
    ):
        check_constant_penalty(penalty, caller=caller)
