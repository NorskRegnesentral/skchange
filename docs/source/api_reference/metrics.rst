.. _metrics:

=======
Metrics
=======
Functions for evaluating detector outputs against ground truth, and helpers
for building scorers that operate on a fitted detector and a dataset.

Metrics follow the uniform signature
``(y_true, y_pred, n_samples=None, ...) -> float``. Length-invariant metrics
accept and ignore ``n_samples``; segmentation-based metrics
(e.g. ``rand_index`` and ``adjusted_rand_index``) require it. Scorers follow the
sklearn-compatible signature ``(detector, X, y=None) -> float``.
Changepoint metrics
-------------------
Compare predicted and true sets of changepoints.

.. currentmodule:: skchange.metrics

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    changepoint_f1_score
    changepoint_precision
    changepoint_recall
    hausdorff_metric
    rand_index
    adjusted_rand_index

Segment anomaly metrics
-----------------------
Compare predicted and true sets of anomalous segments.

.. currentmodule:: skchange.metrics

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    segment_anomaly_f1_score
    segment_anomaly_precision
    segment_anomaly_recall

Scorers
-------
Build and resolve sklearn-compatible scorers from metrics.

.. currentmodule:: skchange.metrics

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    make_detector_scorer
    resolve_scoring
    n_changepoints
    n_segment_anomalies
    n_segments
