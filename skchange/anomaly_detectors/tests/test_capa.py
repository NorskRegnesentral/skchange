"""Tests for CAPA and all available savings."""

import pandas as pd
import pytest

from skchange.anomaly_detectors.capa import Capa
from skchange.anomaly_detectors.mvcapa import Mvcapa
from skchange.anomaly_scores import SAVINGS
from skchange.costs import COSTS
from skchange.datasets.generate import generate_alternating_data

COSTS_AND_SAVINGS = COSTS + SAVINGS


@pytest.mark.parametrize("Saving", COSTS_AND_SAVINGS)
@pytest.mark.parametrize("Detector", [Capa, Mvcapa])
def test_capa_anomalies(Detector, Saving):
    """Test Capa anomalies."""
    n_segments = 2
    seg_len = 20
    df = generate_alternating_data(
        n_segments=n_segments,
        mean=20,
        segment_length=seg_len,
        p=5,
        affected_proportion=0.2,
        random_state=8,
    )
    detector = Detector(
        saving=Saving(),
        collective_penalty_scale=2.0,
        ignore_point_anomalies=True,  # To get test coverage.
    )
    anomalies = detector.fit_predict(df)
    if isinstance(anomalies, pd.DataFrame):
        anomalies = anomalies.iloc[:, 0]
    # End point also included as a changepoint
    assert (
        len(anomalies) == 1
        and anomalies.array.left[0] == seg_len
        and anomalies.array.right[0] == 2 * seg_len - 1
    )
