import numpy as np
import pytest
import seaborn as sns
from scipy.stats import chi2, kstest

from skchange.change_detectors import CROPS, PELT
from skchange.change_scores import to_change_score
from skchange.costs._rank_cost import RankCost
from skchange.datasets import (
    generate_continuous_piecewise_linear_signal,
    generate_piecewise_normal_data,
)


def test_rank_cost_single_variable_no_change():
    # Single variable, no change
    X = np.arange(10).reshape(-1, 1)
    cost = RankCost()
    cost.fit(X)
    starts = np.array([0])
    ends = np.array([10])
    costs = cost._evaluate_optim_param(starts, ends)
    # Should be a single value, negative and not zero
    assert costs.shape == (1, 1)
    assert costs[0, 0] < 0


def test_rank_cost_single_variable_with_change():
    # Single variable, clear change in distribution
    X = np.concatenate([np.random.rand(5), np.random.rand(5) * 10]).reshape(-1, 1)
    cost = RankCost()
    cost.fit(X)
    starts = np.array([0, 0, 5])
    ends = np.array([10, 5, 10])
    costs = cost._evaluate_optim_param(starts, ends)
    # Both segments should have negative costs, but different values
    assert costs.shape == (3, 1)
    assert costs[0, 0] - (costs[1, 0] + costs[2, 0]) > 0


def test_rank_cost_multivariate_no_change():
    # Multivariate, no change
    X = np.tile(np.arange(10), (2, 1)).T
    cost = RankCost()
    cost.fit(X)
    starts = np.array([0])
    ends = np.array([10])
    costs = cost._evaluate_optim_param(starts, ends)
    assert costs.shape == (1, 1)
    assert costs[0, 0] < 0


def test_rank_cost_multivariate_with_change():
    # Multivariate, change in one variable
    X = np.zeros((10, 2))
    X[:5, 0] = 1 * np.random.rand(5)
    X[5:, 0] = 10 * np.random.rand(5)
    X[:, 1] = np.arange(10)
    cost = RankCost()
    cost.fit(X)
    starts = np.array([0, 5])
    ends = np.array([5, 10])
    costs = cost._evaluate_optim_param(starts, ends)
    assert costs.shape == (2, 1)
    assert costs[0, 0] != pytest.approx(costs[1, 0])


def test_rank_cost_multivariate_change_both_vars():
    # Multivariate, change in both variables
    X = np.zeros((10, 2))
    X[:5, 0] = 1
    X[5:, 0] = 10
    X[:5, 1] = 2
    X[5:, 1] = 20
    cost = RankCost()
    cost.fit(X)
    starts = np.array([0, 5])
    ends = np.array([5, 10])
    costs = cost._evaluate_optim_param(starts, ends)
    assert costs.shape == (2, 1)
    assert costs[0, 0] != pytest.approx(costs[1, 0])


def test_rank_cost_min_size_property():
    cost = RankCost()
    assert cost.min_size == 2


def test_rank_cost_model_size():
    cost = RankCost()
    assert cost.get_model_size(3) == 6


def test_rank_cost_on_changing_slope_data():
    lengths = [200, 150, 150]
    changing_mv_gaussian_data = generate_piecewise_normal_data(
        means=[0, 5, 10], variances=[1, 3, 2], lengths=lengths, n_variables=5
    )
    expected_change_points = np.cumsum(lengths)[:-1]

    cost = RankCost()
    change_detector = PELT(cost=cost, min_segment_length=5)
    change_detector.fit(changing_mv_gaussian_data)

    crops_detector = CROPS(
        cost=cost,
        min_segment_length=2,
        min_penalty=1.0e1,
        max_penalty=1.0e3,
        segmentation_selection="elbow",
    )
    crops_detector.fit(changing_mv_gaussian_data)

    pred_crops_change_points = crops_detector.predict(changing_mv_gaussian_data)
    assert (
        np.abs(pred_crops_change_points.ilocs - expected_change_points) < 5
    ).all(), "CROPS change points do not match expected change points"


def test_change_score_distribution():
    # TODO: Test distribution of change score on multivariate Gaussian data:
    # n = 200 samples, with cut point at n/8, n/2, 7*n/8.
    np.random.seed(510)
    cost = RankCost()
    # rank_change_score = to_change_score(cost)

    n_distribution_samples = 500
    data_length = 200

    cut_points = [data_length // 8, data_length // 2, 7 * data_length // 8]
    change_score_samples = np.zeros((n_distribution_samples, len(cut_points)))
    change_score_cuts = [
        np.array([[0, cut_point], [cut_point, data_length]]) for cut_point in cut_points
    ]

    n_variables = 10

    for i in range(n_distribution_samples):
        sample = generate_piecewise_normal_data(
            n_samples=data_length,
            n_variables=n_variables,
            means=[0],
            variances=[1],
            lengths=[data_length],
        )
        # rank_change_score.fit(sample)
        cost.fit(sample)
        for j, change_score_cut in enumerate(change_score_cuts):
            change_score = -cost.evaluate(change_score_cut).sum()
            change_score_samples[i, j] = change_score

    # Use Kolmogorov-Smirnov test to compare to chi2 distribution:
    chi2_at_n_variables_df = chi2(df=n_variables)
    for j, cut_point in enumerate(cut_points):
        res = kstest(change_score_samples[:, j], chi2_at_n_variables_df.cdf)
        assert res.pvalue > 0.01, (
            f"KS test failed for cut at {cut_point}: p={res.pvalue}"
        )
