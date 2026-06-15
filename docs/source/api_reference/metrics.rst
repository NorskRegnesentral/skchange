.. _metrics:

=======
Metrics
=======
Functions for evaluating detector outputs against ground truth, and helpers
for building scorers that operate on a fitted detector and a dataset.

Metrics follow the signature ``(y_true, y_pred) -> float``. Scorers follow the
sklearn-compatible signature ``(detector, X, y=None) -> float``.

Changepoint metrics
-------------------
Compare predicted and true sets of changepoints.

.. currentmodule:: skchange.new_api.metrics

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    changepoint_f1_score
    changepoint_precision
    changepoint_recall
    hausdorff_metric

Segment anomaly metrics
-----------------------
Compare predicted and true sets of anomalous segments.

.. currentmodule:: skchange.new_api.metrics

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    segment_anomaly_f1_score
    segment_anomaly_precision
    segment_anomaly_recall

Segment label metrics
---------------------
Compare predicted and true dense per-sample segment labels.

.. currentmodule:: skchange.new_api.metrics

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    rand_index
    adjusted_rand_index

Scorers
-------
Build and resolve sklearn-compatible scorers from metrics.

.. currentmodule:: skchange.new_api.metrics

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    make_detector_scorer
    resolve_scoring
    n_changepoints
    n_segment_anomalies
    n_segments
