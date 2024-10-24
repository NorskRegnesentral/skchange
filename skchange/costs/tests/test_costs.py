import numpy as np
import pytest

from skchange.costs import VALID_COSTS, cost_factory
from skchange.datasets.generate import generate_alternating_data


@pytest.mark.parametrize("cost", VALID_COSTS)
def test_costs(cost: str):
    """Test all available costs."""
    n = 50
    df = generate_alternating_data(n_segments=1, segment_length=n, p=1, random_state=5)
    cost_f, init_cost_f = cost_factory(cost)
    params = init_cost_f(df.values)
    costs = np.zeros(n)
    starts = np.arange(n - 10)
    ends = np.repeat(n - 1, len(starts))
    costs = cost_f(params, starts, ends=ends)
    assert np.all(costs >= 0.0)


def test_custom_cost():
    """Test custom cost."""
    with pytest.raises(ValueError):
        # Need to be jitted to work.
        # Cannot test jitted function because numba is turned off in CI testing.

        def init_cost_f(X: np.ndarray) -> np.ndarray:
            return X

        def cost_f(params: np.ndarray, start: int, end: int, split: int) -> float:
            return 10.0

        cost_factory((cost_f, init_cost_f))
