"""Metrics for changepoint detection evaluation.

Metrics are organised into three submodules matching the native output types of
``BaseChangeDetector``:

- **_changepoint** — metrics that compare changepoint index arrays
  (output of ``predict_changepoints()``): ``hausdorff_metric``,
  ``changepoint_f1_score``.
- **_segment_label** — metrics that compare dense per-sample label arrays
  (output of ``predict()``): ``rand_index``, ``adjusted_rand_index``.
- **_segment_anomaly** — metrics that compare anomalous-interval arrays of shape
  ``(n_anomalies, 2)`` (output of ``predict_segment_anomalies()``):
  ``segment_anomaly_f1_score``.

All public functions are re-exported here for convenience::

    from skchange.new_api.metrics import hausdorff_metric, rand_index
"""

from skchange.new_api.metrics._changepoint import (
    changepoint_f1_score,
    changepoint_precision,
    changepoint_recall,
    hausdorff_metric,
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
]
