"""Tests for CAPA and all available savings."""

import numpy as np
import pandas as pd
import pytest

from skchange.anomaly_detectors import CAPA, MVCAPA
from skchange.anomaly_detectors.capa import run_capa
from skchange.anomaly_scores import SAVINGS, Saving, to_saving
from skchange.compose import PenalisedScore
from skchange.costs import COSTS, BaseCost, L2Cost, MultivariateGaussianCost
from skchange.costs.tests.test_all_costs import find_fixed_param_combination
from skchange.datasets.generate import generate_alternating_data
from skchange.penalties import ChiSquarePenalty
from skchange.utils.validation.enums import EvaluationType

COSTS_AND_SAVINGS = COSTS + SAVINGS


@pytest.mark.parametrize("Saving", COSTS_AND_SAVINGS)
@pytest.mark.parametrize("Detector", [CAPA, MVCAPA])
def test_capa_anomalies(Detector, Saving):
    """Test CAPA anomalies."""
    saving = Saving.create_test_instance()
    if isinstance(saving, BaseCost):
        fixed_params = find_fixed_param_combination(Saving)
        saving = saving.set_params(**fixed_params)

    if Detector is MVCAPA and saving.evaluation_type == EvaluationType.MULTIVARIATE:
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


def test_capa_errors():
    """Test CAPA error cases."""
    cost = MultivariateGaussianCost([0.0, np.eye(2)])

    # Test point saving must have a minimum size of 1
    with pytest.raises(ValueError):
        CAPA(point_saving=cost)

    # Test min_segment_length must be greater than 2
    with pytest.raises(ValueError):
        CAPA(min_segment_length=1)

    # Test max_segment_length must be greater than min_segment_length
    with pytest.raises(ValueError):
        CAPA(min_segment_length=5, max_segment_length=4)


def test_capa_different_data_shapes():
    """Test CAPA with segment and point savings having different data shapes."""

    # Create detector
    detector = CAPA()
    detector.fit(pd.DataFrame(np.random.randn(10, 2)))

    # Create two PenalisedScore objects with different data shapes
    segment_saving = to_saving(L2Cost(param=0.0))
    point_saving = to_saving(L2Cost(param=0.0))

    segment_penalty = ChiSquarePenalty()
    point_penalty = ChiSquarePenalty()

    segment_data = pd.DataFrame(np.random.randn(20, 2))
    point_data = pd.DataFrame(np.random.randn(30, 2))  # Different number of samples

    segment_penalised_saving = PenalisedScore(segment_saving, segment_penalty)
    point_penalised_saving = PenalisedScore(point_saving, point_penalty)

    segment_penalised_saving.fit(segment_data)
    point_penalised_saving.fit(point_data)

    # Test that run_capa raises ValueError due to different shapes
    with pytest.raises(ValueError, match="same number of samples"):
        run_capa(
            segment_penalised_saving=segment_penalised_saving,
            point_penalised_saving=point_penalised_saving,
            min_segment_length=2,
            max_segment_length=10,
        )
