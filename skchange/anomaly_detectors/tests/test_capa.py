"""Tests for CAPA and all available savings."""

import numpy as np
import pandas as pd
import pytest

from skchange.anomaly_detectors.capa import CAPA
from skchange.anomaly_detectors.mvcapa import (
    Mvcapa,
    capa_penalty_factory,
    combined_mvcapa_penalty,
    dense_mvcapa_penalty,
    intermediate_mvcapa_penalty,
    sparse_mvcapa_penalty,
)
from skchange.anomaly_scores import SAVINGS, Saving
from skchange.costs import COSTS, BaseCost, GaussianCovCost
from skchange.costs.tests.test_all_costs import find_fixed_param_combination
from skchange.datasets.generate import generate_alternating_data

COSTS_AND_SAVINGS = COSTS + SAVINGS


@pytest.mark.parametrize("Saving", COSTS_AND_SAVINGS)
@pytest.mark.parametrize("Detector", [CAPA, Mvcapa])
def test_capa_anomalies(Detector, Saving):
    """Test CAPA anomalies."""
    saving = Saving.create_test_instance()
    if isinstance(saving, BaseCost):
        fixed_params = find_fixed_param_combination(Saving)
        saving = saving.set_params(**fixed_params)

    if Detector is Mvcapa and saving.evaluation_type == "multivariate":
        # Mvcapa requires univariate saving.
        pytest.skip("Skipping test for Mvcapa with multivariate saving.")

    n_segments = 2
    seg_len = 20
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
        collective_saving=saving,
        collective_penalty_scale=2.0,
        min_segment_length=p + 1,
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


def test_mvcapa_errors():
    """Test Mvcapa error cases."""
    cov_mat = np.eye(2)
    cost = GaussianCovCost([0.0, cov_mat])
    saving = Saving(cost)

    # Test collective saving must be univariate
    with pytest.raises(ValueError):
        Mvcapa(collective_saving=saving)

    # Test point saving must have a minimum size of 1
    with pytest.raises(ValueError):
        Mvcapa(point_saving=saving)

    # Test min_segment_length must be greater than 2
    with pytest.raises(ValueError):
        Mvcapa(min_segment_length=1)

    # Test max_segment_length must be greater than min_segment_length
    with pytest.raises(ValueError):
        Mvcapa(min_segment_length=5, max_segment_length=4)


def test_capa_penalty_factory():
    """Test capa_penalty_factory with different penalties."""
    n, p, n_params_per_variable, scale = 100, 5, 1, 1.0

    # Test dense penalty
    penalty_func = capa_penalty_factory("dense")
    alpha, betas = penalty_func(n, p, n_params_per_variable, scale)
    assert penalty_func == dense_mvcapa_penalty

    # Test sparse penalty
    penalty_func = capa_penalty_factory("sparse")
    alpha, betas = penalty_func(n, p, n_params_per_variable, scale)
    assert penalty_func == sparse_mvcapa_penalty

    # Test intermediate penalty
    penalty_func = capa_penalty_factory("intermediate")
    alpha, betas = penalty_func(n, p, n_params_per_variable, scale)
    assert penalty_func == intermediate_mvcapa_penalty

    # Test combined penalty
    penalty_func = capa_penalty_factory("combined")
    alpha, betas = penalty_func(n, p, n_params_per_variable, scale)
    assert penalty_func == combined_mvcapa_penalty

    # Test unknown penalty
    with pytest.raises(ValueError):
        capa_penalty_factory("unknown")
