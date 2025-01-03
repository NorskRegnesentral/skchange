"""Tests for CAPA and all available savings."""

import numpy as np
import pandas as pd
import pytest

from skchange.anomaly_detectors import CAPA, MVCAPA
from skchange.anomaly_scores import SAVINGS, Saving
from skchange.costs import COSTS, BaseCost, MultivariateGaussianCost
from skchange.costs.tests.test_all_costs import find_fixed_param_combination
from skchange.datasets.generate import generate_alternating_data

COSTS_AND_SAVINGS = COSTS + SAVINGS


@pytest.mark.parametrize("Saving", COSTS_AND_SAVINGS)
@pytest.mark.parametrize("Detector", [CAPA, MVCAPA])
def test_capa_anomalies(Detector, Saving):
    """Test CAPA anomalies."""
    saving = Saving.create_test_instance()
    if isinstance(saving, BaseCost):
        fixed_params = find_fixed_param_combination(Saving)
        saving = saving.set_params(**fixed_params)

    if Detector is MVCAPA and saving.evaluation_type == "multivariate":
        # MVCAPA requires univariate saving.
        pytest.skip("Skipping test for MVCAPA with multivariate saving.")

    n_segments = 2
    seg_len = 50
    p = 5
    df = generate_alternating_data(
        n_segments=n_segments,
        mean=20,
        segment_length=seg_len,
        p=p,
        affected_proportion=0.2,
        random_state=8,
    )
    detector = Detector(
        segment_saving=saving,
        min_segment_length=20,
        ignore_point_anomalies=True,  # To get test coverage.
    )
    anomalies = detector.fit_predict(df)
    if isinstance(anomalies, pd.DataFrame):
        anomalies = anomalies.iloc[:, 0]
    # End point also included as a changepoint
    assert (
        len(anomalies) == 1
        and anomalies.array.left[0] == seg_len
        and anomalies.array.right[0] == 2 * seg_len
    )


@pytest.mark.parametrize("Detector", [CAPA, MVCAPA])
def test_capa_anomalies_segment_length(Detector):
    detector = Detector.create_test_instance()
    min_segment_length = 5
    detector.set_params(
        segment_penalty=0.0,
        min_segment_length=min_segment_length,
    )

    n = 100
    df = generate_alternating_data(n_segments=1, segment_length=n, random_state=13)
    anomalies = detector.fit_predict(df)["ilocs"]

    anomaly_lengths = anomalies.array.right - anomalies.array.left
    assert np.all(anomaly_lengths == 5)


@pytest.mark.parametrize("Detector", [CAPA, MVCAPA])
def test_capa_point_anomalies(Detector):
    detector = Detector.create_test_instance()
    n_segments = 2
    seg_len = 50
    p = 3
    df = generate_alternating_data(
        n_segments=n_segments,
        mean=20,
        segment_length=seg_len,
        p=p,
        random_state=134,
    )
    point_anomaly_iloc = 20
    df.iloc[point_anomaly_iloc] += 50

    anomalies = detector.fit_predict(df)
    estimated_point_anomaly_iloc = anomalies["ilocs"].iloc[0]

    assert point_anomaly_iloc == estimated_point_anomaly_iloc.left


def test_mvcapa_errors():
    """Test MVCAPA error cases."""
    cov_mat = np.eye(2)
    cost = MultivariateGaussianCost([0.0, cov_mat])
    saving = Saving(cost)

    # Test segment saving must be univariate
    with pytest.raises(ValueError):
        MVCAPA(segment_saving=saving)

    # Test point saving must have a minimum size of 1
    with pytest.raises(ValueError):
        MVCAPA(point_saving=saving)

    # Test min_segment_length must be greater than 2
    with pytest.raises(ValueError):
        MVCAPA(min_segment_length=1)

    # Test max_segment_length must be greater than min_segment_length
    with pytest.raises(ValueError):
        MVCAPA(min_segment_length=5, max_segment_length=4)
