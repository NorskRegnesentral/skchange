"""Tests for MultivariateGaussianSaving with valued baseline parameters."""

import numpy as np

from skchange.conftest import (
    CHANGEPOINT,
    make_no_change_X,
    make_single_change_X,
)
from skchange.detectors import CAPA
from skchange.interval_scorers import MultivariateGaussianSaving
from skchange.interval_scorers._costs.multivariate_gaussian_cost import (
    _multivariate_gaussian_precompute,
)
from skchange.interval_scorers._savings.multivariate_gaussian_saving import (
    _multivariate_gaussian_cost_fixed,
    _multivariate_gaussian_cost_fixed_cached,
)

# Baseline parameters — must match data-generation parameters in the sanity tests.
BASELINE_MEAN = 0.0  # scalar; broadcast to all features
BASELINE_COV = 1.0  # scalar; broadcast to identity matrix
LOC_AFTER = 10.0  # Mean of the second segment; should differ from BASELINE_MEAN.


def _make_capa(saving):
    # min_segment_length=5 prevents spurious 2-3-sample detections from
    # random fluctuations in multivariate data.
    return CAPA(segment_saving=saving, min_segment_length=5)


# ---------------------------------------------------------------------------
# Baseline attribute tests
# ---------------------------------------------------------------------------


def test_multivariate_gaussian_saving_stores_scalar_baseline():
    """Scalar mean and cov are broadcast to (p,) and (p,p) identity after fit."""
    scorer = MultivariateGaussianSaving(
        baseline_mean=BASELINE_MEAN, baseline_cov=BASELINE_COV
    )
    X = make_no_change_X(scorer, n_features=2, loc=BASELINE_MEAN)
    scorer.fit(X)
    np.testing.assert_array_equal(scorer.baseline_mean_, [BASELINE_MEAN, BASELINE_MEAN])
    np.testing.assert_array_almost_equal(scorer.baseline_cov_, BASELINE_COV * np.eye(2))


def test_multivariate_gaussian_saving_accepts_explicit_matrix():
    """An explicit SPD covariance matrix is accepted and stored after fit."""
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    scorer = MultivariateGaussianSaving(
        baseline_mean=np.array([1.0, 2.0]), baseline_cov=cov
    )
    X = make_no_change_X(scorer, n_features=2, loc=1.0)
    scorer.fit(X)
    np.testing.assert_array_equal(scorer.baseline_mean_, [1.0, 2.0])
    np.testing.assert_array_almost_equal(scorer.baseline_cov_, cov)


# ---------------------------------------------------------------------------
# CAPA sanity tests with valued baselines
# ---------------------------------------------------------------------------


def test_capa_multivariate_gaussian_saving_finds_no_changepoint():
    """CAPA with matched multivariate Gaussian valued baseline finds no changepoints."""
    scorer = MultivariateGaussianSaving(
        baseline_mean=BASELINE_MEAN, baseline_cov=BASELINE_COV
    )
    X = make_no_change_X(scorer, n_features=2, loc=BASELINE_MEAN)
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict(X)
    assert len(cpts) == 0, f"Expected 0 changepoints, got {len(cpts)}: {cpts}"


def test_capa_multivariate_gaussian_saving_finds_single_changepoint():
    """CAPA with matched multivariate Gaussian baseline detects the changepoint."""
    scorer = MultivariateGaussianSaving(
        baseline_mean=BASELINE_MEAN, baseline_cov=BASELINE_COV
    )
    X = make_single_change_X(
        scorer, n_features=2, loc_before=BASELINE_MEAN, loc_after=LOC_AFTER
    )
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict(X)
    assert len(cpts) == 1, f"Expected 1 changepoint, got {len(cpts)}: {cpts}"
    assert (
        abs(cpts[0] - CHANGEPOINT) <= 6
    ), f"Changepoint {cpts[0]} is too far from true changepoint {CHANGEPOINT}."


def test_cached_fixed_cost_handles_undersized_intervals():
    """Cached fixed Gaussian cost returns inf for short intervals and keeps looping."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(8, 2))
    cache = _multivariate_gaussian_precompute(X, store_cov=True)

    starts = np.array([0, 0], dtype=np.int64)
    ends = np.array([2, 8], dtype=np.int64)
    min_size = 3
    mean = np.zeros(2)
    inv_cov = np.eye(2)
    log_det_cov = 0.0

    cached_costs = _multivariate_gaussian_cost_fixed_cached(
        starts,
        ends,
        cache["feature_sums"],
        cache["outer_product_sums"],
        mean,
        log_det_cov,
        inv_cov,
        min_size,
    )
    uncached_costs = _multivariate_gaussian_cost_fixed(
        starts, ends, X, mean, log_det_cov, inv_cov, min_size
    )

    assert np.isinf(cached_costs[0, 0])
    assert np.isfinite(cached_costs[1, 0])
    np.testing.assert_allclose(cached_costs, uncached_costs)
