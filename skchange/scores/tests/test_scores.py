"""Tests for all available changepoint scores."""

import numpy as np
import pytest

from skchange.datasets.generate import generate_teeth_data
from skchange.scores.score_factory import VALID_CHANGE_SCORES, score_factory


@pytest.mark.parametrize("score", VALID_CHANGE_SCORES)
def test_scores(score):
    """Test all available changepoint scores."""
    n = 50
    df = generate_teeth_data(n_segments=1, segment_length=n, p=1, random_state=5)
    score_f, init_score_f = score_factory(score)
    params = init_score_f(df.values)
    scores = np.zeros(n)
    for splits in np.arange(10, n - 10).reshape(-1, 1):
        score_values = score_f(
            params, starts=np.array([0]), ends=np.array([49]), splits=splits
        )
        assert isinstance(score_values, np.ndarray)
        scores[splits] = score_values

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


def test_mean_cov_score_negative_definite_error():
    """Test that mean_cov_score raises an error.

    When the covariance matrix is not positive definite, the
    'mean_cov' score should raise a RuntimeError.
    """
    n = 50
    x = np.zeros((n, 1))
    score_f, init_score_f = score_factory("mean_cov")
    with pytest.raises(RuntimeError):
        params = init_score_f(x)
        score_f(
            params, starts=np.array([0]), ends=np.array([49]), splits=np.array([25])
        )
