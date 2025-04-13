.. _penalties:

Penalties
=========
Penalties govern the trade-off between the number of change points and the fit of the
model. They are used by all detectors in ``skchange``.

Base
----

.. currentmodule:: skchange.penalties.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BasePenalty

Constant penalties
------------------
The penalty for each additional change point in the model is constant.

.. currentmodule:: skchange.penalties

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ConstantPenalty
    BICPenalty
    ChiSquarePenalty


Linear penalties
------------------
The penalty for each additional change point in the model is linear in the number of
variables affected by the change.
Only relevant for multivariate data and detectors supporting variable identification.

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    LinearPenalty
    LinearChiSquarePenalty

Nonlinear penalties
------------------
The penalty for each additional change point in the model is non-linear in the number of
variables affected by the change.
Only relevant for multivariate data and detectors supporting variable identification.

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    NonlinearPenalty
    NonlinearChiSquarePenalty

Composition
-----------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MinimumPenalty


Utility functions
-----------------

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    as_penalty
