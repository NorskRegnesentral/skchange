import numpy as np
import pandas as pd
import pytest

from skchange.change_scores import CUSUM
from skchange.penalties import (
    ConstantPenalty,
    LinearChiSquarePenalty,
    MinimumPenalty,
    NonlinearChiSquarePenalty,
)

df = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
scorer = CUSUM()


def test_minimum_penalty_initialization():
    penalty1 = ConstantPenalty.create_test_instance()
    penalty2 = LinearChiSquarePenalty.create_test_instance()
    penalty = MinimumPenalty([penalty1, penalty2], scale=1.0)
    penalty.fit(df, scorer)
    assert penalty.penalty_type == "linear"
    assert penalty.p_ == df.shape[1]


def test_minimum_penalty_invalid_initialization():
    with pytest.raises(ValueError):
        MinimumPenalty([], scale=1.0)


def test_minimum_penalty_base_values():
    penalty1 = ConstantPenalty.create_test_instance()
    penalty2 = LinearChiSquarePenalty.create_test_instance()
    penalty = MinimumPenalty([penalty1, penalty2], scale=1.0)
    penalty.fit(df, scorer)
    expected_base_values = np.minimum(penalty1._base_values, penalty2._base_values)
    np.testing.assert_array_equal(penalty._base_values, expected_base_values)


def test_minimum_penalty_values():
    penalty1 = ConstantPenalty.create_test_instance()
    penalty2 = LinearChiSquarePenalty.create_test_instance()
    penalty = MinimumPenalty([penalty1, penalty2], scale=2.0)
    penalty.fit(df, scorer)
    expected_values = 2.0 * np.minimum(penalty1._base_values, penalty2._base_values)
    np.testing.assert_array_equal(penalty.values, expected_values)


def test_minimum_penalty_constant():
    penalty1 = ConstantPenalty(10)
    penalty2 = ConstantPenalty(5)
    penalty = MinimumPenalty([penalty1, penalty2], scale=1.0)
    penalty.fit(df, scorer)
    assert penalty.penalty_type == "constant"


def test_minimum_penalty_nonlinear():
    penalty1 = NonlinearChiSquarePenalty.create_test_instance()
    penalty2 = ConstantPenalty(5)
    penalty = MinimumPenalty([penalty1, penalty2], scale=1.0)
    penalty.fit(df, scorer)
    assert penalty.penalty_type == "nonlinear"
    assert penalty.p_ == df.shape[1]
