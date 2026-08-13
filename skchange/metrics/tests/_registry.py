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
    The metric function under test. Called as
    ``func(true, pred, **kwargs)`` by the contract tests.
``true`` : array-like
    A representative ground-truth input.
``pred_different`` : array-like
    A prediction that meaningfully differs from ``true``, used to verify the
    output range.  Should *not* be identical to ``true``.
``perfect_value`` : float
    The expected return value of ``func(true, true.copy(), **kwargs)``.
    Typically ``0.0`` for lower-is-better metrics, ``1.0`` for
    higher-is-better.
``lower_better`` : bool
    ``True`` for metrics where 0.0 is the best score (e.g. Hausdorff distance).
    Controls which direction the range test checks.
``n_samples`` : int, optional
    Length of the underlying time series. Set on every metric case so the
    uniform ``(true, pred, n_samples=...)`` signature is exercised.
    Length-invariant metrics accept and ignore it; ``rand_index`` and
    ``adjusted_rand_index`` require it. Forwarded to ``func`` as a keyword
    argument.
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
_N_SAMPLES = 50  # covers the maximum index in the changepoint arrays above

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
        "n_samples": _N_SAMPLES,
    },
    {
        "id": "changepoint_precision",
        "func": changepoint_precision,
        "true": _CHANGEPOINTS_TRUE,
        "pred_different": _CHANGEPOINTS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
        "n_samples": _N_SAMPLES,
    },
    {
        "id": "changepoint_recall",
        "func": changepoint_recall,
        "true": _CHANGEPOINTS_TRUE,
        "pred_different": _CHANGEPOINTS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
        "n_samples": _N_SAMPLES,
    },
    {
        "id": "changepoint_f1_score",
        "func": changepoint_f1_score,
        "true": _CHANGEPOINTS_TRUE,
        "pred_different": _CHANGEPOINTS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
        "n_samples": _N_SAMPLES,
    },
    {
        "id": "rand_index",
        "func": rand_index,
        "true": _CHANGEPOINTS_TRUE,
        "pred_different": _CHANGEPOINTS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
        "n_samples": _N_SAMPLES,
    },
    {
        "id": "adjusted_rand_index",
        "func": adjusted_rand_index,
        "true": _CHANGEPOINTS_TRUE,
        "pred_different": _CHANGEPOINTS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
        "n_samples": _N_SAMPLES,
    },
    {
        "id": "segment_anomaly_precision",
        "func": segment_anomaly_precision,
        "true": _INTERVALS_TRUE,
        "pred_different": _INTERVALS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
        "n_samples": _N_SAMPLES,
    },
    {
        "id": "segment_anomaly_recall",
        "func": segment_anomaly_recall,
        "true": _INTERVALS_TRUE,
        "pred_different": _INTERVALS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
        "n_samples": _N_SAMPLES,
    },
    {
        "id": "segment_anomaly_f1_score",
        "func": segment_anomaly_f1_score,
        "true": _INTERVALS_TRUE,
        "pred_different": _INTERVALS_DIFF,
        "perfect_value": 1.0,
        "lower_better": False,
        "n_samples": _N_SAMPLES,
    },
]
