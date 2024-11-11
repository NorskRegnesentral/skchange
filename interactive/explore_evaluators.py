"""Interactive exploration of the evaluators module."""

from skchange.datasets.generate import generate_alternating_data
from skchange.interval_evaluators.costs import L2Cost


def generate_all_intervals(df):
    """Generate all possible intervals."""
    n = len(df)
    return [[i, j] for i in range(n) for j in range(i + 1, n + 1)]


df = generate_alternating_data(
    n_segments=10, mean=10, segment_length=200, p=1, random_state=2
)
cost = L2Cost()
cost.fit(df)

intervals = generate_all_intervals(df)
cost.evaluate(intervals)
