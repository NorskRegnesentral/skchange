"""Test instances for metrics in ``skchange.new_api.metrics``.

Design
------
Common contract tests in ``test_all.py`` are parametrized over
``METRIC_TEST_CASES`` — one dict per metric function.  Adding a new metric
to the common test suite requires only adding a new entry here; no changes
to ``test_all.py`` are needed.

Each entry has the following keys:

``id`` : str
    Human-readable name used as the pytest test ID.
``func`` : callable
    The metric function under test.
``true`` : array-like
    A representative ground-truth input.
``pred_different`` : array-like
    A prediction that meaningfully differs from ``true``, used to verify the
    output range.  Should *not* be identical to ``true``.
``perfect_value`` : float
    The expected return value of ``func(true, true.copy())``.  Typically ``0.0``
    for lower-is-better metrics, ``1.0`` for higher-is-better.
``lower_better`` : bool
    ``True`` for metrics where 0.0 is the best score (e.g. Hausdorff distance).
    Controls which direction the range test checks.
``requires_equal_length`` : bool, optional (default ``False``)
    Set to ``True`` for metrics that require ``len(true) == len(pred)``
    (e.g. ``rand_index``).  Tests that pass mismatched-length inputs will be
    skipped for these metrics.
"""

import numpy as np

from skchange.new_api.metrics import (
    adjusted_rand_index,
    changepoint_f1_score,
    changepoint_precision,
    changepoint_recall,
    hausdorff_metric,
    rand_index,
    segment_anomaly_f1_score,
    segment_anomaly_precision,
    segment_anomaly_recall,
)

# ---------------------------------------------------------------------------
# Test data shared across metric types
# ---------------------------------------------------------------------------

_CHANGEPOINTS_TRUE = np.array([10, 20, 30])
_CHANGEPOINTS_DIFF = np.array([11, 20, 29])

_LABELS_TRUE = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
_LABELS_DIFF = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1])

_INTERVALS_TRUE = np.array([[10, 20], [40, 50]])
_INTERVALS_DIFF = np.array([[30, 35]])


METRIC_TEST_CASES = [
    {
        "id": "hausdorff_metric",
        "func": hausdorff_metric,
        "true": _CHANGEPOINTS_TRUE,
        "pred_different": _CHANGEPOINTS_DIFF,
        "perfect_value": 0.0,
        "lower_better": True,
    },
    {
        "id": "changepoint_precision",
        "func": changepoint_precision,
        "true": _CHANGEPOINTS_TRUE,
        "pred_different": _CHANGEPOINTS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
    },
    {
        "id": "changepoint_recall",
        "func": changepoint_recall,
        "true": _CHANGEPOINTS_TRUE,
        "pred_different": _CHANGEPOINTS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
    },
    {
        "id": "changepoint_f1_score",
        "func": changepoint_f1_score,
        "true": _CHANGEPOINTS_TRUE,
        "pred_different": _CHANGEPOINTS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
    },
    {
        "id": "rand_index",
        "func": rand_index,
        "true": _LABELS_TRUE,
        "pred_different": _LABELS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
        "requires_equal_length": True,
    },
    {
        "id": "adjusted_rand_index",
        "func": adjusted_rand_index,
        "true": _LABELS_TRUE,
        "pred_different": _LABELS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
        "requires_equal_length": True,
    },
    {
        "id": "segment_anomaly_precision",
        "func": segment_anomaly_precision,
        "true": _INTERVALS_TRUE,
        "pred_different": _INTERVALS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
    },
    {
        "id": "segment_anomaly_recall",
        "func": segment_anomaly_recall,
        "true": _INTERVALS_TRUE,
        "pred_different": _INTERVALS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
    },
    {
        "id": "segment_anomaly_f1_score",
        "func": segment_anomaly_f1_score,
        "true": _INTERVALS_TRUE,
        "pred_different": _INTERVALS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
    },
]
