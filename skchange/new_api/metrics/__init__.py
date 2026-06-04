"""Metrics and scorers for changepoint detection evaluation.

- **Metrics** compare ground truth and prediction: ``(y_true, y_pred) -> float``.
- **Scorers** evaluate a fitted detector on data: ``(detector, X, y=None) -> float``.
"""

from skchange.new_api.metrics._changepoint import (
    changepoint_f1_score,
    changepoint_precision,
    changepoint_recall,
    hausdorff_metric,
)
from skchange.new_api.metrics._scorer import (
    make_detector_scorer,
    n_changepoints,
    n_segment_anomalies,
    n_segments,
    resolve_scoring,
)
from skchange.new_api.metrics._segment_anomaly import (
    segment_anomaly_f1_score,
    segment_anomaly_precision,
    segment_anomaly_recall,
)
from skchange.new_api.metrics._segment_label import adjusted_rand_index, rand_index

__all__ = [
    "hausdorff_metric",
    "changepoint_precision",
    "changepoint_recall",
    "changepoint_f1_score",
    "rand_index",
    "adjusted_rand_index",
    "segment_anomaly_precision",
    "segment_anomaly_recall",
    "segment_anomaly_f1_score",
    "n_changepoints",
    "n_segment_anomalies",
    "n_segments",
    "make_detector_scorer",
    "resolve_scoring",
]
