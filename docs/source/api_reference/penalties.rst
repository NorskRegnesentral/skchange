.. _penalties:

Penalties
=========
Penalties govern the trade-off between the number of change points and the fit of the
model. They are used by all current detectors in ``skchange``. Utility functions are
provided for helping to create commonly used penalties.


Constant penalties
------------------
The penalty for each additional change point in the model is constant.

.. currentmodule:: skchange.penalties

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    make_bic_penalty
    make_chi2_penalty


Linear penalties
------------------
The penalty for each additional change point in the model is linear in the number of
variables affected by the change. Only relevant for multivariate data.Some detectors
use such penalties to identify the variables responsible for the change or anomaly.
Penalised scores using linear penalties are faster to compute than non-linear penalties.

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    make_linear_penalty
    make_linear_chi2_penalty

Nonlinear penalties
------------------
The penalty for each additional change point in the model is non-linear in the number of
variables affected by the change. Only relevant for multivariate data. Some detectors
use such penalties to identify the variables responsible for the change or anomaly.

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    make_nonlinear_chi2_penalty
