import numpy as np
import pytest

from skchange.costs import VALID_SAVINGS, saving_factory
from skchange.datasets.generate import generate_alternating_data


@pytest.mark.parametrize("savings", VALID_SAVINGS)
def test_savings(savings):
    """Test all available savings."""
    n = 50
    df = generate_alternating_data(n_segments=1, segment_length=n, p=1, random_state=5)
    savings_f, init_savings_f = saving_factory(savings)
    params = init_savings_f(df.values)
    savings_values = np.zeros(n)
    starts = np.arange(n - 10)
    ends = np.repeat(n - 1, len(starts))
    savings_values = savings_f(params, starts, ends=ends)
    assert np.all(savings_values >= 0.0)


def test_custom_savings():
    """Test custom savings."""
    # No longer need to be jitted to work.
    # Cannot test jitted function because numba is turned off in CI testing.

    def init_savings_f(X: np.ndarray) -> np.ndarray:
        return X

    def savings_f(params: np.ndarray, start: int, end: int, split: int) -> float:
        return 10.0

    assert (savings_f, init_savings_f) == saving_factory((savings_f, init_savings_f))
