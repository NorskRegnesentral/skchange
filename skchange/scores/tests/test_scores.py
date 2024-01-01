"""Tests for all available changepoint scores."""

import numpy as np
import pytest

from skchange.datasets.generate import teeth
from skchange.scores.score_factory import VALID_CHANGE_SCORES, score_factory


@pytest.mark.parametrize("score", VALID_CHANGE_SCORES)
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
    with pytest.raises(ValueError):
        # Need to be jitted to work.
        # Cannot test jitted function because numba is turned of in CI testing.

        def init_score_f(X: np.ndarray) -> np.ndarray:
            return X

        def score_f(params: np.ndarray, start: int, end: int, split: int) -> float:
            return 10.0

        score_factory((score_f, init_score_f))
