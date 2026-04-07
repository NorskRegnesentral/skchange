"""Common contract tests for all metrics in ``skchange.new_api.metrics``."""

import math

import numpy as np
import pytest

from skchange.new_api.metrics.tests._registry import METRIC_TEST_CASES

_all_metrics = pytest.mark.parametrize(
    "case", METRIC_TEST_CASES, ids=[c["id"] for c in METRIC_TEST_CASES]
)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@_all_metrics
def test_metric_returns_float(case):
    """All metrics must return a plain Python float."""
    result = case["func"](case["true"], case["true"].copy())
    assert type(result) is float


# ---------------------------------------------------------------------------
# ArrayLike inputs
# ---------------------------------------------------------------------------


@_all_metrics
def test_metric_accepts_list_inputs(case):
    """Metrics must accept plain Python lists as inputs."""
    true_list = case["true"].tolist()
    result = case["func"](true_list, true_list)
    assert isinstance(result, float)


@_all_metrics
def test_metric_accepts_numpy_inputs(case):
    """Metrics must accept numpy arrays as inputs."""
    result = case["func"](case["true"], case["true"].copy())
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Perfect prediction
# ---------------------------------------------------------------------------


@_all_metrics
def test_metric_perfect_prediction(case):
    """metric(x, x) must equal the expected perfect value."""
    result = case["func"](case["true"], case["true"].copy())
    assert result == pytest.approx(case["perfect_value"])


# ---------------------------------------------------------------------------
# Output range
# ---------------------------------------------------------------------------


@_all_metrics
def test_metric_output_in_range(case):
    """Output must be in [0, 1] for bounded metrics, >= 0 for Hausdorff."""
    result = case["func"](case["true"], case["pred_different"])
    if case["lower_better"]:
        # Hausdorff: non-negative
        assert result >= 0.0
    else:
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Input mutation
# ---------------------------------------------------------------------------


@_all_metrics
def test_metric_does_not_mutate_inputs(case):
    """Metrics must not modify their input arrays."""
    true_copy = case["true"].copy()
    pred = case["true"].copy()
    case["func"](case["true"], pred)
    np.testing.assert_array_equal(case["true"], true_copy)
    np.testing.assert_array_equal(pred, true_copy)


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


@_all_metrics
def test_metric_both_empty_returns_defined(case):
    """metric(empty, empty) must return a finite float without raising."""
    empty = case["empty"]
    result = case["func"](empty, empty)
    assert isinstance(result, float)
    assert math.isfinite(result)


@_all_metrics
def test_metric_one_empty_returns_defined(case):
    """metric(true, empty) and metric(empty, pred) must return a float without raising.

    Skipped for metrics that require equal-length inputs (e.g. rand_index).
    """
    if case.get("requires_equal_length", False):
        pytest.skip("metric requires equal-length inputs")
    empty = case["empty"]
    result_true_only = case["func"](case["true"], empty)
    result_pred_only = case["func"](empty, case["true"].copy())
    assert isinstance(result_true_only, float)
    assert isinstance(result_pred_only, float)
