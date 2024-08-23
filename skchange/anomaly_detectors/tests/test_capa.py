"""Tests for CAPA and all available savings."""

import pandas as pd
import pytest

from skchange.anomaly_detectors.capa import Capa
from skchange.anomaly_detectors.mvcapa import Mvcapa
from skchange.costs.saving_factory import VALID_SAVINGS
from skchange.datasets.generate import generate_teeth_data


@pytest.mark.parametrize("saving", VALID_SAVINGS)
@pytest.mark.parametrize("detector_class", [Capa, Mvcapa])
def test_capa_anomalies(detector_class, saving):
    """Test Capa anomalies."""
    n_segments = 2
    seg_len = 20
    df = generate_teeth_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=5, random_state=8
    )
    detector = detector_class(
        saving=saving,
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
