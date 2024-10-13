.. _base:

Base
====

The :mod:`skchange.base` module contains the abstract detector class
:class:`skchange.base.BaseDetector`. This class serves as a minimalistic template for all the
detectors in `sktime`.

For common detection tasks, like changepoint detection or anomaly detection, there
are dedicated subclasses of :class:`BaseDetector` that implement a stricter template
for these tasks. These subclasses are located in the respective modules, e.g.
:mod:`skchange.change_detectors` or :mod:`skchange.anomaly_detectors`.


.. currentmodule:: skchange.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseDetector
