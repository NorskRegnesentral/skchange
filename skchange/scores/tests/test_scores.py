"""Tests for all available changepoint scores."""

import numpy as np
import pytest
from numba import njit

from skchange.datasets.generate import teeth
from skchange.scores.score_factory import VALID_SCORES, score_factory


@pytest.mark.parametrize("score", VALID_SCORES)
def test_scores(score):
    """Test all available changepoint scores."""
    n = 50
    df = teeth(n_segments=1, segment_length=n, p=1, random_state=5)
    score_f, init_score_f = score_factory(score)
    params = init_score_f(df.values)
    scores = np.zeros(n)
    for split in range(10, n - 10):
        score_value = score_f(params, start=0, end=49, split=split)
        assert isinstance(score_value, float)
        scores[split] = score_value

    assert np.all(scores >= 0.0)


def test_custom_score():
    """Test custom score."""
    n = 50
    df = teeth(n_segments=1, segment_length=n, p=1, random_state=5)
    with pytest.raises(ValueError):

        def init_score_f(X: np.ndarray) -> np.ndarray:
            return X

        def score_f(params: np.ndarray, start: int, end: int, split: int) -> float:
            return 10.0

        score_factory(score_f, init_score_f)

    @njit
    def jit_init_score_f(X: np.ndarray) -> np.ndarray:
        return X

    @njit
    def jit_score_f(params: np.ndarray, start: int, end: int, split: int) -> float:
        return 10.0

    score_f, init_score_f = score_factory((jit_score_f, jit_init_score_f))
    params = init_score_f(df.values)
    score_value = score_f(params, start=0, end=49, split=25)
    assert isinstance(score_value, float)
    assert np.all(score_value >= 0.0)
