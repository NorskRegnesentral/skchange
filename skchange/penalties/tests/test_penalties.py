"""Tests for all penalties."""

import numpy as np
import pandas as pd
import pytest

from skchange.change_scores import CUSUM
from skchange.penalties import PENALTIES, BasePenalty

df = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
scorer = CUSUM()


@pytest.mark.parametrize("Penalty", PENALTIES)
def test_penalty_type(Penalty: BasePenalty):
    penalty = Penalty.create_test_instance()
    supported_penalty_types = ["constant", "linear", "nonlinear"]
    assert penalty.penalty_type in supported_penalty_types


@pytest.mark.parametrize("Penalty", PENALTIES)
def test_fit(Penalty: BasePenalty):
    penalty = Penalty.create_test_instance()
    penalty.fit(df, scorer)
    assert penalty._is_fitted


@pytest.mark.parametrize("Penalty", PENALTIES)
def test_values(Penalty: BasePenalty):
    penalty = Penalty.create_test_instance()
    penalty.fit(df, scorer)

    assert isinstance(penalty.values, np.ndarray)

    if penalty.penalty_type == "constant":
        assert penalty.values.shape == (1,)
    else:
        assert penalty.values.shape == (penalty.p,)

    assert np.all(penalty.values >= 0.0)

    assert np.all(np.diff(penalty.values) >= 0)


@pytest.mark.parametrize("Penalty", PENALTIES)
def test_scale(Penalty: BasePenalty):
    penalty = Penalty.create_test_instance()
    assert isinstance(penalty.scale, (int, float))
    assert penalty.scale > 0

    with pytest.raises(ValueError):
        penalty.set_params(scale=-1.0)
