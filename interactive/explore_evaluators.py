"""Interactive exploration of the evaluators module."""

import numpy as np
import pandas as pd

from skchange.change_detectors import PELT
from skchange.costs import L2Cost
from skchange.datasets import generate_alternating_data


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

# Test PELT
df = generate_alternating_data(
    n_segments=5, mean=10, segment_length=100000, p=1, random_state=2
)
n = 10000
df = pd.DataFrame(np.random.normal(0, 1, size=n))
cost = L2Cost()
detector = PELT(cost=cost)
detector.fit_predict(df)
