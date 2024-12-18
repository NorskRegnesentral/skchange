import numpy as np
import pytest

from skchange.penalties import (
    ConstantPenalty,
    LinearChiSquarePenalty,
    MinimumPenalty,
    NonlinearChiSquarePenalty,
)


def test_minimum_penalty_initialization():
    penalty1 = ConstantPenalty(10)
    penalty2 = LinearChiSquarePenalty(100, 10, 1)
    penalty = MinimumPenalty([penalty1, penalty2], scale=1.0)
    assert penalty.penalty_type == "linear"
    assert penalty.p == 10


def test_minimum_penalty_invalid_initialization():
    penalty1 = ConstantPenalty(10)
    with pytest.raises(ValueError):
        MinimumPenalty([penalty1], scale=1.0)


def test_minimum_penalty_base_values():
    penalty1 = ConstantPenalty(10)
    penalty2 = LinearChiSquarePenalty(100, 10, 1)
    penalty = MinimumPenalty([penalty1, penalty2], scale=1.0)
    expected_base_values = np.minimum(penalty1.base_values, penalty2.base_values)
    np.testing.assert_array_equal(penalty.base_values, expected_base_values)


def test_minimum_penalty_values():
    penalty1 = ConstantPenalty(10)
    penalty2 = LinearChiSquarePenalty(100, 10, 1)
    penalty = MinimumPenalty([penalty1, penalty2], scale=2.0)
    expected_values = 2.0 * np.minimum(penalty1.base_values, penalty2.base_values)
    np.testing.assert_array_equal(penalty.values, expected_values)


def test_minimum_penalty_constant():
    penalty1 = ConstantPenalty(10)
    penalty2 = ConstantPenalty(5)
    penalty = MinimumPenalty([penalty1, penalty2], scale=1.0)
    assert penalty.penalty_type == "constant"
    assert penalty.p == 1


def test_minimum_penalty_nonlinear():
    penalty1 = NonlinearChiSquarePenalty(100, 10, 1)
    penalty2 = ConstantPenalty(5)
    penalty = MinimumPenalty([penalty1, penalty2], scale=1.0)
    assert penalty.penalty_type == "nonlinear"
    assert penalty.p == 10


def test_minimum_penalty_unequal_dims():
    penalty1 = LinearChiSquarePenalty(100, 5, 1)
    penalty2 = NonlinearChiSquarePenalty(100, 10, 1)
    with pytest.raises(ValueError, match="same number of variables"):
        MinimumPenalty([penalty1, penalty2], scale=1.0)
