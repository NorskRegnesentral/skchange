.. _interval_scorers:

================
Interval scorers
================
Interval scorers evaluate a scalar score on intervals of a time series. They
are the building blocks of all detectors in Skchange and come in several
flavours: costs, change scores, savings, and transient scores. All scorers
share the common base class
:class:`~skchange.interval_scorers.BaseIntervalScorer`.

Base classes
------------
.. currentmodule:: skchange.interval_scorers

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseIntervalScorer
    BaseCost
    BaseChangeScore
    BaseSaving
    BaseTransientScore

Costs
-----
A cost measures how well a single interval is fit by a parametric model.

.. currentmodule:: skchange.interval_scorers

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    EDFCost
    GaussianCost
    L1Cost
    L2Cost
    LaplaceCost
    LinearRegressionCost
    LinearTrendCost
    MultivariateGaussianCost
    MultivariateTCost
    PoissonCost
    RankCost

Change scores
-------------
A change score measures the evidence for a change between two adjacent
intervals. :class:`CostChangeScore` adapts any cost into a change score.

.. currentmodule:: skchange.interval_scorers

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ContinuousLinearTrendScore
    CostChangeScore
    CUSUM
    ESACScore
    MultivariateGaussianScore
    RankScore

Savings
-------
A saving measures the evidence that an interval deviates from a baseline
model. Savings are used by segment-anomaly detectors such as CAPA.

.. currentmodule:: skchange.interval_scorers

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    GaussianSaving
    L1Saving
    L2Saving
    LaplaceSaving
    LinearRegressionSaving
    LinearTrendSaving
    MultivariateGaussianSaving
    MultivariateTSaving
    PoissonSaving

Transient scores
----------------
A transient score measures the evidence for a short-lived deviation inside an
interval. Transient scores are used by detectors that target point anomalies
embedded in a longer segment. :class:`CostTransientScore` adapts any cost
into a transient score.

.. currentmodule:: skchange.interval_scorers

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CostTransientScore
    L2TransientScore

Predicates
----------
Type predicates for runtime checks on scorer instances.

.. currentmodule:: skchange.interval_scorers

.. autosummary::
    :toctree: auto_generated/
    :template: functions.rst

    is_cost
    is_change_score
    is_saving
    is_transient_score
    is_aggregated_score
    is_penalised_score
