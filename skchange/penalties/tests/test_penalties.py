"""Tests for all penalties."""

import numpy as np
import pytest

from skchange.penalties import PENALTIES, BasePenalty


@pytest.mark.parametrize("Penalty", PENALTIES)
def test_penalty_type(Penalty: BasePenalty):
    penalty = Penalty.create_test_instance()
    supported_penalty_types = ["constant", "linear", "nonlinear"]
    assert penalty.penalty_type in supported_penalty_types


@pytest.mark.parametrize("Penalty", PENALTIES)
def test_values(Penalty: BasePenalty):
    penalty = Penalty.create_test_instance()
    assert isinstance(penalty.base_values, np.ndarray)
    assert isinstance(penalty.values, np.ndarray)

    if penalty.penalty_type == "constant":
        assert penalty.base_values.shape == (1,)
        assert penalty.values.shape == (1,)
    else:
        assert penalty.base_values.shape == (penalty.p,)
        assert penalty.values.shape == (penalty.p,)

    assert np.all(penalty.base_values >= 0.0)
    assert np.all(penalty.values >= 0.0)


@pytest.mark.parametrize("Penalty", PENALTIES)
def test_scale(Penalty: BasePenalty):
    penalty = Penalty.create_test_instance()
    assert isinstance(penalty.scale, (int, float))
    assert penalty.scale > 0
