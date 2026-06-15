.. _coding_guidelines:

=================
Coding guidelines
=================
Skchange follows the
`scikit-learn developer guidelines <https://scikit-learn.org/stable/developers/develop.html>`_
as closely as possible. This page collects the conventions that are most
relevant when writing or reviewing code for Skchange, with links to the
scikit-learn documentation for the full rationale.

Formatting and linting
----------------------
Formatting and linting are handled by `Ruff <https://docs.astral.sh/ruff/>`_,
configured under ``[tool.ruff]`` in ``pyproject.toml``. The same checks run via
``pre-commit`` locally and in the continuous integration pipeline.

Key settings:

* Line length is **88 characters**, matching the
  `Black <https://black.readthedocs.io/>`_ default.
* Imports are sorted with ``isort`` rules.
* Docstrings are checked with ``pydocstyle`` using the **numpy** convention.

Run all formatting and lint checks locally with:

.. code-block:: bash

    pre-commit run --all-files

Naming conventions
------------------
Skchange follows
`scikit-learn naming conventions <https://scikit-learn.org/stable/developers/develop.html#coding-guidelines>`_:

* ``snake_case`` for modules, functions, variables, and parameters.
* ``CamelCase`` for class names.
* ``UPPER_CASE`` for module-level constants.
* Private helpers and modules are prefixed with a single underscore
  (e.g. ``_utils.py``, ``_compute_score``).
* Attributes that are **set during** ``fit`` end with a trailing underscore
  (e.g. ``self.threshold_``, ``self.changepoints_``). This mirrors the
  `scikit-learn estimated attribute convention
  <https://scikit-learn.org/stable/developers/develop.html#estimated-attributes>`_
  and is how callers tell fitted state apart from user-supplied parameters.

Common variable names align with scikit-learn:

* ``X`` is the input data of shape ``(n_samples, n_features)``.
* ``y`` is the target or label array. It is rarely used in Skchange's unsupervised setting.
* ``n_samples`` is the number of timepoints.
* ``n_features`` is the number of variables or channels.

Estimator API conventions
-------------------------
Detectors, costs, and other estimators in Skchange follow the
`scikit-learn estimator API
<https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects>`_.
The most important rules are:

* **No work in** ``__init__``. ``__init__`` only stores the constructor
  arguments verbatim on ``self``, with **no validation or transformation**.
  See
  `scikit-learn instantiation rules
  <https://scikit-learn.org/stable/developers/develop.html#instantiation>`_.
* **Parameter names match attribute names.** Every constructor argument
  ``foo`` is stored as ``self.foo`` so that ``get_params`` /
  ``set_params`` and cloning work correctly.
* **All learning happens in** ``fit``. Input validation, fitted attributes
  (ending in ``_``), and ``self`` is returned. See
  `scikit-learn fitting
  <https://scikit-learn.org/stable/developers/develop.html#fitting>`_.
* ``predict`` and friends are **stateless**: they return values and do
  not modify ``self``.

Detectors may expose any of the following ``predict_*`` methods, depending on
what the algorithm computes.

* ``predict(X)`` returns an ndarray of shape ``(n_samples,)`` containing dense
  per-sample integer segment labels. It is part of the universal interface and
  is always implemented.
* ``predict_changepoints(X)`` returns an ndarray of shape
  ``(n_changepoints,)`` containing sorted start indices of segments. It
  is part of the universal interface and is always implemented.
* ``predict_segment_anomalies(X)`` returns an ndarray of shape
  ``(n_anomalies, 2)`` containing start (inclusive) and end (exclusive) indices of
  anomalous segments. It is only implemented on detectors that identify anomalous
  segments.
* ``predict_scores(X, return_index=False)`` returns the detector's internal
  scoring objective as a 1D ndarray whose length is detector-specific. When
  ``return_index=True``, it returns the tuple ``(scores, index_dict)`` where
  the dictionary carries algorithm-specific metadata that locates each score
  on the timeline.
* ``predict_all(X)`` is a convenience method that returns all outputs
  computed in a single pass as a dictionary. The keys are detector-specific
  and are not a stable cross-detector contract.

Skchange intentionally deviates from scikit-learn in a few respects to fit the
time series context. Most notably, Skchange detectors are not expected to be invariant
to the order of the samples, so they fail the ``check_methods_sample_order_invariance``
check.

Docstrings
----------
All public modules, classes, functions, and methods must have a docstring
written in the
`numpydoc style <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
``numpydoc`` validation is enabled in the documentation build, so missing
or malformed sections will be flagged.

A minimal example::

    def my_function(x, threshold=0.5):
        """Summary line in imperative mood.

        Extended description, optional.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The input data.
        threshold : float, default=0.5
            Description of ``threshold``.

        Returns
        -------
        result : ndarray of shape (n_samples,)
            Description of the return value.
        """

Guidelines:

* Start with a one-line summary in the imperative mood
  ("Compute ...", not "Computes ...").
* Document **every** parameter and return value with type and shape.
* Use ``default=<value>`` rather than ``optional`` for parameters that
  have a default, matching scikit-learn.
* Cross-reference other Skchange objects with
  ``:class:`~skchange.detectors.PELT``` or
  ``:func:`~skchange.metrics.f1_score```.
* Include a ``References`` section with citations and an ``Examples``
  section with a runnable doctest where it adds value.

Tests
-----
Tests live next to the code they cover, in a ``tests/`` subpackage
(e.g. ``skchange/detectors/tests/``). Naming follows
`pytest conventions <https://docs.pytest.org/en/stable/explanation/goodpractices.html>`_.

* Test files are named ``test_*.py``.
* Test functions are named ``test_*``.
* Use ``pytest.mark.parametrize`` for varying inputs rather than loops.

Each subpackage also contains a shared contract test module that exercises
every implementation in that subpackage against a common set of API checks.
These modules are named ``test_all.py`` in the new API
(e.g. ``skchange/new_api/detectors/tests/test_all.py``,
``skchange/new_api/interval_scorers/tests/test_all.py``,
``skchange/new_api/metrics/tests/test_all.py``). The set of instances they run
against is declared in a sibling ``_registry.py`` file. When you add a new
detector, scorer, or metric, register a representative set of instances in the
corresponding ``_registry.py`` so the shared checks are exercised against your
implementation automatically.

These contract tests cover the constructor contract, ``fit`` and ``predict``
round trips, the fitted-attribute naming convention, cloning, and other
estimator-level invariants. They are inspired by
`scikit-learn's check_estimator
<https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html>`_.

Run the full test suite locally with:

.. code-block:: bash

    pytest

Type hints
----------
Type hints are encouraged for new public functions but are not required
everywhere. When adding hints, prefer ``numpy.typing`` aliases (e.g.
``npt.NDArray[np.floating]``) and standard ``typing`` constructs over
custom protocols.

Further reading
---------------
* `scikit-learn developer guide
  <https://scikit-learn.org/stable/developers/index.html>`_
* `scikit-learn API design
  <https://scikit-learn.org/stable/developers/develop.html>`_
* `numpydoc style guide
  <https://numpydoc.readthedocs.io/en/latest/format.html>`_
* `Ruff rules <https://docs.astral.sh/ruff/rules/>`_
