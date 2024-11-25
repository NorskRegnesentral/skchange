import numpy as np
import pytest

from skchange.change_scores import CHANGE_SCORES
from skchange.datasets.generate import generate_alternating_data


@pytest.mark.parametrize("ChangeScore", CHANGE_SCORES)
def test_scores(ChangeScore):
    """Test all available changepoint scores."""
    n = 50
    df = generate_alternating_data(n_segments=1, segment_length=n, p=2, random_state=5)
    change_score = ChangeScore.create_test_instance()
    change_score.fit(df)
    scores = np.zeros(n)
    splits = np.arange(10, n - 10, dtype=int).reshape(-1, 1)
    intervals = np.concatenate(
        [
            np.zeros(splits.shape, dtype=int),
            splits,
            np.full(splits.shape, n, dtype=int),
        ],
        axis=1,
    )
    scores = change_score.evaluate(intervals)
    assert np.all(scores >= 0.0)
