.. _tuning:

======
Tuning
======
Hyperparameter tuning and penalty calibration utilities for detectors.

Calibrated detector
-------------------

.. currentmodule:: skchange.tuning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CalibratedDetector

Penalty calibration
-------------------

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    calibrate_penalty_scale
    penalty_curve
    unpenalised_scores

Null samplers
-------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseNullSampler
    BlockBootstrapSampler
    GaussianSampler
    PermutationSampler
