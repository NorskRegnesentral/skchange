"""Tests for CAPA and all available savings."""

import pytest

from skchange.anomaly_detectors.capa import Capa
from skchange.anomaly_detectors.mvcapa import Mvcapa
from skchange.costs.saving_factory import VALID_SAVINGS
from skchange.datasets.generate import generate_teeth_data


@pytest.mark.parametrize("saving", VALID_SAVINGS)
def test_capa_anomalies(saving):
    """Test Capa anomalies."""
    n_segments = 2
    seg_len = 20
    df = generate_teeth_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=5, random_state=8
    )
    for detector_class in [Capa, Mvcapa]:
        detector = detector_class(
            saving=saving, fmt="sparse", collective_penalty_scale=2.0
        )
        anomalies = detector.fit_predict(df)
        # End point also included as a changepoint
        assert (
            len(anomalies) == 1
            and anomalies.loc[0, "start"] == seg_len
            and anomalies.loc[0, "end"] == 2 * seg_len - 1
        )
