"""Tests for non-linear penalties."""

import numpy as np
import pytest

from skchange.change_scores import CUSUM
from skchange.penalties import (
    NonlinearChiSquarePenalty,
    NonlinearPenalty,
)


def test_nonlinear_penalty_init():
    with pytest.raises(TypeError):
        NonlinearPenalty([1, 2, 3])
    with pytest.raises(ValueError):
        NonlinearPenalty(np.array([[1, 2], [3, 4]]))
    with pytest.raises(ValueError):
        NonlinearPenalty(np.array([]))
    with pytest.raises(ValueError):
        NonlinearPenalty(np.array([-1, 2, 3]))
    with pytest.raises(ValueError):
        NonlinearPenalty(np.array([3, 2, 1]))

    penalty = NonlinearPenalty(base_values=np.array([1, 2, 3]))
    assert penalty.base_values.size == 3


def test_nonlinear_penalty_fit():
    penalty = NonlinearPenalty(base_values=np.array([1, 2, 3]))
    X = np.random.rand(100, 3)
    scorer = CUSUM()
    penalty._fit(X, scorer)
    assert penalty.base_values_.size == 3


def test_nonlinear_penalty_base_values():
    penalty = NonlinearPenalty(base_values=np.array([1, 2, 3]))
    X = np.random.rand(100, 3)
    scorer = CUSUM()
    penalty.fit(X, scorer)
    assert np.array_equal(penalty._base_values, np.array([1, 2, 3]))


def test_nonlinear_chisquare_penalty_fit():
    penalty = NonlinearChiSquarePenalty(scale=1.0)
    X = np.random.rand(100, 3)
    scorer = CUSUM()
    penalty.fit(X, scorer)
    assert penalty._base_penalty_values.size == 3


def test_nonlinear_chisquare_too_many_params():
    penalty = NonlinearChiSquarePenalty(scale=1.0)
    X = np.random.rand(100, 3)
    scorer = CUSUM()
    penalty.n_params_per_variable_ = 3
    with pytest.raises(ValueError):
        penalty._fit(X, scorer)
