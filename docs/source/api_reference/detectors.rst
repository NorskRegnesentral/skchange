.. _detectors:

=========
Detectors
=========
Detectors operate on a single (univariate or multivariate) time series and
segment it into homogeneous regions. All detectors inherit from
:class:`~skchange.new_api.detectors.BaseChangeDetector` and expose the
universal ``predict`` method, which returns a sorted numpy array of
changepoint indices. Some detectors additionally expose
``predict_segment_anomalies``, ``predict_scores`` or ``predict_all``,
depending on what the underlying algorithm computes.

Base
----
.. currentmodule:: skchange.new_api.detectors

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseChangeDetector

Changepoint detectors
---------------------
Detectors that implement ``predict``.

.. currentmodule:: skchange.new_api.detectors

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CROPS
    MovingWindow
    PELT
    SeededBinarySegmentation

Segment anomaly detectors
-----------------------------------------
Detectors that additionally implement ``predict_segment_anomalies``.

.. currentmodule:: skchange.new_api.detectors

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CAPA
    CircularBinarySegmentation

Utilities
---------
.. currentmodule:: skchange.new_api.detectors

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    is_change_detector
