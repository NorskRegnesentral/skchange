.. _penalties:

=========
Penalties
=========
Penalties govern the trade-off between the number of changepoints in the
model and the fit of the model. They are used by all detectors in Skchange.
Helper functions are provided for the commonly used penalty shapes.

Constant penalties
------------------
The penalty for each additional changepoint is constant.

.. currentmodule:: skchange.penalties

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    bic_penalty
    chi2_penalty

Linear penalties
----------------
The penalty for each additional changepoint is linear in the number of
variables affected by the change. Only relevant for multivariate data. Some
detectors use such penalties to identify the variables responsible for the
change or anomaly. Penalised scores using linear penalties are faster to
compute than non-linear penalties.

.. currentmodule:: skchange.penalties

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    linear_penalty
    linear_chi2_penalty

Nonlinear penalties
-------------------
The penalty for each additional changepoint is non-linear in the number of
variables affected by the change. Only relevant for multivariate data.

.. currentmodule:: skchange.penalties

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    nonlinear_chi2_penalty
    mvcapa_penalty
