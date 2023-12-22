"""Tests for CAPA and all available savings."""

import pytest

from skchange.anomaly_detectors.capa import Capa
from skchange.costs.saving_factory import VALID_SAVINGS
from skchange.datasets.generate import teeth


@pytest.mark.parametrize("saving", VALID_SAVINGS)
def test_capa_anomalies(saving):
    """Test Capa anomalies."""
    n_segments = 2
    seg_len = 20
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=8
    )
    detector = Capa(saving=saving, fmt="sparse", collective_penalty_scale=2.0)
    anomalies = detector.fit_predict(df)
    # End point also included as a changepoint
    assert (
        len(anomalies) == 1
        and anomalies[0].left == seg_len
        and anomalies[0].right == 2 * seg_len - 1
    )
