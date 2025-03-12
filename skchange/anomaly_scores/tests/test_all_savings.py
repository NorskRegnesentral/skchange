import numpy as np
import pytest

from skchange.anomaly_scores import SAVINGS, to_saving
from skchange.costs import COSTS
from skchange.datasets import generate_alternating_data

SCORES_AND_COSTS = SAVINGS + COSTS


@pytest.mark.parametrize("Saving", SAVINGS)
def test_savings_positive(Saving):
    """Test all available savings."""
    n = 50
    df = generate_alternating_data(n_segments=1, segment_length=n, p=1, random_state=5)
    saving = to_saving(Saving.create_test_instance())
    saving.fit(df)

    starts = np.arange(n - 10)
    ends = np.repeat(n - 1, len(starts))
    intervals = np.column_stack((starts, ends))
    saving_values = saving.evaluate(intervals)

    assert np.all(saving_values >= 0.0)
