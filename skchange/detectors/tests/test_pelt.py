"""Dedicated tests for :class:`PELT` computational correctness.

These tests target the PELT-specific dynamic-programming contracts that the
generic registry-based contract tests in ``test_all.py`` do not cover:
pruning equivalence to optimal partitioning, agreement across
``min_segment_length`` values, the dense-changepoints corner with zero
penalty, cross-validation against the ``ruptures`` reference implementation,
and step-size (jump) PELT behaviour.
"""

import numpy as np
import pytest
import ruptures as rpt
from sklearn.base import clone

from skchange.datasets import generate_piecewise_normal_data
from skchange.detectors import PELT
from skchange.detectors._crops import _evaluate_segmentation
from skchange.detectors._pelt import _run_pelt
from skchange.interval_scorers import L2Cost, L2Saving, MultivariateGaussianCost

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

# 5-segment alternating mean sequence used by parametric PELT correctness tests.
ALTERNATING_SEQUENCE = generate_piecewise_normal_data(
    means=[0.0, 10.5],
    variances=0.5,
    lengths=21,
    n_segments=5,
    n_variables=1,
    seed=5,
)

# 30-segment version of the same alternating signal used to cross-check with
# ruptures over longer signals.
LONG_ALTERNATING_SEQUENCE = generate_piecewise_normal_data(
    means=[0.0, 10.5],
    variances=0.5,
    lengths=21,
    n_segments=30,
    n_variables=1,
    seed=5,
)


@pytest.fixture
def cost() -> L2Cost:
    """Fresh L2 cost for each test."""
    return L2Cost()


@pytest.fixture
def penalty() -> float:
    """BIC-style penalty for the alternating sequence length."""
    return 2.0 * np.log(len(ALTERNATING_SEQUENCE))


def _partition_cost(
    X: np.ndarray,
    changepoints: np.ndarray,
    cost: L2Cost,
    cache: dict,
    penalty: float,
) -> float:
    """Penalised cost of a segmentation: ``sum_seg_costs + n_changepoints * penalty``.

    Matches the additive convention used by PELT's cumulative optimal costs.
    """
    boundaries = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.asarray(changepoints, dtype=np.int64),
            np.array([X.shape[0]], dtype=np.int64),
        )
    )
    intervals = np.column_stack((boundaries[:-1], boundaries[1:]))
    seg_cost = float(np.sum(cost.evaluate(cache, intervals)))
    return seg_cost + penalty * len(changepoints)


# ---------------------------------------------------------------------------
# Pruning correctness
# ---------------------------------------------------------------------------


def test_pelt_with_and_without_pruning_is_the_same(cost: L2Cost, penalty: float):
    """PELT with pruning must match optimal partitioning (no pruning)."""
    X = ALTERNATING_SEQUENCE
    cost.fit(X)

    opt_part = _run_pelt(cost, X, penalty=penalty, min_segment_length=1, prune=False)
    pelt = _run_pelt(cost, X, penalty=penalty, min_segment_length=1, prune=True)

    np.testing.assert_array_equal(pelt.changepoints, opt_part.changepoints)
    np.testing.assert_array_almost_equal(pelt.optimal_costs, opt_part.optimal_costs)


# ---------------------------------------------------------------------------
# Parametric correctness vs. optimal partitioning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_segment_length", [1, 5, 10])
@pytest.mark.parametrize(
    "signal_end_index", list(range(20, len(ALTERNATING_SEQUENCE) + 1, 5))
)
def test_pelt_on_tricky_data(
    cost: L2Cost,
    penalty: float,
    min_segment_length: int,
    signal_end_index: int,
):
    """PELT must agree with optimal partitioning across many segment lengths and
    truncated signal lengths, and the final cumulative cost must match the
    recomputed penalised partition cost."""
    X = ALTERNATING_SEQUENCE[:signal_end_index]
    cost.fit(X)
    cache = cost.precompute(X)

    pelt = _run_pelt(
        cost, X, penalty=penalty, min_segment_length=min_segment_length, prune=True
    )
    opt_part = _run_pelt(
        cost, X, penalty=penalty, min_segment_length=min_segment_length, prune=False
    )

    np.testing.assert_array_equal(pelt.changepoints, opt_part.changepoints)
    np.testing.assert_array_almost_equal(
        pelt.optimal_costs, opt_part.optimal_costs, decimal=10
    )
    np.testing.assert_almost_equal(
        pelt.optimal_costs[-1],
        _partition_cost(X, pelt.changepoints, cost, cache, penalty),
        decimal=10,
        err_msg="PELT cumulative cost does not match recomputed partition cost.",
    )


# ---------------------------------------------------------------------------
# Dense changepoints with zero penalty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_segment_length", [1, 2, 5, 10])
def test_pelt_dense_changepoints_parametrized(cost: L2Cost, min_segment_length: int):
    """With penalty=0, changepoints must land at every ``min_segment_length`` index."""
    seg_len = 50
    X = np.linspace(0, seg_len, seg_len).reshape(-1, 1)
    cost.fit(X)

    pelt = _run_pelt(cost, X, penalty=0.0, min_segment_length=min_segment_length)

    expected = [
        i * min_segment_length for i in range(1, X.shape[0] // min_segment_length)
    ]
    np.testing.assert_array_equal(pelt.changepoints, expected)


# ---------------------------------------------------------------------------
# Cross-validation against ruptures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_segment_length", [33, 39])
def test_pelt_matches_ruptures(cost: L2Cost, penalty: float, min_segment_length: int):
    """Skchange PELT must agree with ``ruptures.Pelt`` and ``ruptures.Dynp`` on
    the long alternating sequence, with identical objective values."""
    X = LONG_ALTERNATING_SEQUENCE
    cost.fit(X)
    cache = cost.precompute(X)
    n = X.shape[0]

    opt_part = _run_pelt(
        cost, X, penalty=penalty, min_segment_length=min_segment_length, prune=False
    )
    pelt = _run_pelt(
        cost, X, penalty=penalty, min_segment_length=min_segment_length, prune=True
    )

    def _objective(cpts: np.ndarray) -> float:
        return _evaluate_segmentation(cost, cache, cpts, n) + len(cpts) * penalty

    pelt_value = _objective(pelt.changepoints)
    opt_part_value = _objective(opt_part.changepoints)

    rpt_dynp_cpts = np.array(
        rpt.Dynp(model="l2", min_size=min_segment_length, jump=1)
        .fit(X)
        .predict(n_bkps=len(opt_part.changepoints))[:-1]
    )
    rpt_pelt_cpts = np.array(
        rpt.Pelt(model="l2", min_size=min_segment_length, jump=1).fit_predict(
            X, pen=penalty
        )[:-1]
    )

    np.testing.assert_array_equal(pelt.changepoints, opt_part.changepoints)
    np.testing.assert_array_equal(pelt.changepoints, rpt_dynp_cpts)
    np.testing.assert_array_equal(pelt.changepoints, rpt_pelt_cpts)

    assert pelt_value == opt_part_value
    assert pelt_value == _objective(rpt_dynp_cpts)
    assert pelt_value == _objective(rpt_pelt_cpts)
    assert abs(pelt.optimal_costs[-1] - opt_part.optimal_costs[-1]) < 1e-12


# ---------------------------------------------------------------------------
# Step-size (jump) PELT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step_size", [3, 5, 10])
def test_pelt_with_step_size(cost: L2Cost, penalty: float, step_size: int):
    """Jump-PELT changepoints must be multiples of ``step_size``, match
    ``ruptures.Pelt(jump=step_size)``, and the final cumulative cost must equal
    the recomputed penalised partition cost."""
    X = ALTERNATING_SEQUENCE
    detector = PELT(cost=cost, step_size=step_size, penalty=penalty)

    cpts = detector.fit_predict(X)
    result = detector.predict_all(X)

    assert len(cpts) > 0
    assert np.all(cpts % step_size == 0)

    rpt_cpts = np.array(
        rpt.Pelt(model="l2", min_size=step_size, jump=step_size).fit_predict(
            X, pen=penalty
        )[:-1]
    )
    np.testing.assert_array_equal(cpts, rpt_cpts)

    cache = detector.cost_.precompute(X)
    expected_value = _partition_cost(X, cpts, detector.cost_, cache, penalty)
    np.testing.assert_allclose(
        result["cumulative_optimal_costs"][-1], expected_value, atol=1e-10
    )


def test_jump_pelt_with_fewer_samples_than_twice_step_size_returns_no_cpts():
    """When ``n_samples < 2 * step_size`` jump-PELT has no room to split."""
    X = np.random.default_rng(0).standard_normal((10, 1))
    detector = PELT(cost=L2Cost(), step_size=6, penalty=1.0)
    cpts = detector.fit_predict(X)
    np.testing.assert_array_equal(cpts, np.array([], dtype=cpts.dtype))


# --------------------------------------------------------------------------- #
# penalty_scale parameter
# --------------------------------------------------------------------------- #
def _make_X(n=80, p=2, seed=0):
    return np.random.default_rng(seed).normal(size=(n, p))


def test_pelt_default_penalty_scale_is_one():
    assert PELT().penalty_scale == 1.0


def test_pelt_penalty_scale_in_get_params():
    det = PELT(penalty_scale=2.5)
    assert det.get_params()["penalty_scale"] == 2.5


def test_pelt_penalty_scale_set_params_roundtrip():
    det = PELT()
    det.set_params(penalty_scale=3.0)
    assert det.penalty_scale == 3.0


def test_pelt_penalty_scale_clone():
    det = PELT(penalty_scale=1.8)
    cloned = clone(det)
    assert cloned.get_params()["penalty_scale"] == 1.8


def test_pelt_penalty_scale_multiplies_base_default_penalty():
    """PELT(penalty_scale=c) must give penalty_ = c * default_penalty."""
    X = _make_X()
    base = PELT(penalty_scale=1.0).fit(X)
    base_penalty = base.penalty_

    for c in [0.5, 2.0, 5.0]:
        det = PELT(penalty_scale=c).fit(X)
        assert det.penalty_ == pytest.approx(c * base_penalty, rel=1e-10)


def test_pelt_penalty_scale_multiplies_explicit_penalty():
    """With an explicit penalty, penalty_ = penalty_scale * penalty."""
    X = _make_X()
    for c in [0.5, 2.0]:
        det = PELT(penalty=4.0, penalty_scale=c).fit(X)
        assert det.penalty_ == pytest.approx(c * 4.0, rel=1e-10)


def test_pelt_penalty_scale_one_matches_legacy():
    """penalty_scale=1 must reproduce current behaviour (no-op)."""
    X = _make_X()
    old = PELT().fit(X)
    new = PELT(penalty_scale=1.0).fit(X)
    assert old.penalty_ == new.penalty_
    np.testing.assert_array_equal(old.predict(X), new.predict(X))


def test_pelt_penalty_scale_changes_detections():
    """Large penalty_scale should suppress detections; small should increase them."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (60, 1)), rng.normal(5, 1, (60, 1))])
    tight = PELT(penalty_scale=10.0).fit(X)
    loose = PELT(penalty_scale=0.1).fit(X)
    n_tight = len(tight.predict(X))
    n_loose = len(loose.predict(X))
    assert n_tight <= n_loose


# ---------------------------------------------------------------------------
# fit() ValueError contracts
# ---------------------------------------------------------------------------


class _PenalisedCost(L2Cost):
    """L2Cost with penalised=True to exercise the allow_penalised=False guard."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.penalised = True
        return tags


def test_pelt_fit_rejects_non_cost_scorer():
    X = np.random.default_rng(0).standard_normal((50, 1))
    with pytest.raises(ValueError, match="cost"):
        PELT(cost=L2Saving()).fit(X)


def test_pelt_fit_rejects_penalised_cost():
    X = np.random.default_rng(0).standard_normal((50, 1))
    with pytest.raises(ValueError, match="cost"):
        PELT(cost=_PenalisedCost()).fit(X)


def test_pelt_fit_rejects_min_segment_length_below_cost_min_size():
    # MultivariateGaussianCost has min_size = n_features + 1 = 4 on 3-feature data.
    X = np.random.default_rng(0).standard_normal((50, 3))
    with pytest.raises(ValueError, match="min_segment_length"):
        PELT(cost=MultivariateGaussianCost(), min_segment_length=1).fit(X)


def test_pelt_fit_rejects_min_segment_length_greater_than_step_size():
    X = np.random.default_rng(0).standard_normal((50, 1))
    with pytest.raises(ValueError, match="min_segment_length"):
        PELT(step_size=2, min_segment_length=5).fit(X)
