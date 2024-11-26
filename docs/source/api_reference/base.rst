.. _base:

Base
====

The :mod:`skchange.base` module contains the abstract detector class
:class:`skchange.base.BaseDetector`. This class serves as a minimalistic template for
all the detectors in :mod:`skchange`.

For common detection tasks, like changepoint detection or anomaly detection, there
are dedicated subclasses of :class:`BaseDetector` that implement a stricter template
for these tasks. These subclasses are located in the respective modules, e.g.
:mod:`skchange.change_detectors.base` or :mod:`skchange.anomaly_detectors.base`.

The :mod:`skchange.base` module also contains the abstract interval scorer class
:class:`skchange.base.BaseIntervalScorer`. This is a common base class for detector
components like cost functions, change scores, and anomaly scores. It serves the
purpose of evaluating different types of scores efficiently over many sets of intervals.

Base classes
------------

.. currentmodule:: skchange.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseDetector
    BaseIntervalScorer
