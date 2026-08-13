.. _utils:

=========
Utilities
=========
General-purpose helpers used across Skchange: tag classes that declare
estimator capabilities, input and parameter validators, and segmentation
conversions.

Tags
----
Dataclasses that declare detector and interval-scorer capabilities.

.. currentmodule:: skchange.utils

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SkchangeTags
    SkchangeInputTags
    ChangeDetectorTags
    IntervalScorerTags

Validation
----------
Input and parameter validation helpers.

.. currentmodule:: skchange.utils

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    validate_data
    check_interval_scorer
    check_interval_specs
    check_penalty

Segmentation
------------
Conversions between dense per-sample segment labels and sparse changepoint
indices.

.. currentmodule:: skchange.utils

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    changepoints_to_labels
    labels_to_changepoints
