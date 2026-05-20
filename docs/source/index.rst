.. _home:

===================
Welcome to skchange
===================

A python library for fast change point and segment anomaly detection.

**Breaking changes expected.** skchange is undergoing a significant API redesign in upcoming releases.
See `Issue #120 <https://github.com/NorskRegnesentral/skchange/issues/120>`_ and the
`migration guide <https://github.com/NorskRegnesentral/skchange/blob/main/skchange/new_api/MIGRATION_GUIDE.md>`_ for details.

- **New API (recommended)** is previewed in ``skchange.new_api.*`` and becomes the default in 0.17.0, when the same names move to top-level (``skchange.detectors``, ``skchange.interval_scorers``, ``skchange.penalties``, ...). Drop ``new_api.`` from imports when upgrading.
- **Current API** (``skchange.change_detectors``, ``skchange.costs``, ...) emits a ``FutureWarning`` in 0.16.x and is removed in 0.17.0.

If you need stability and the old `sktime <https://www.sktime.net/>`_ compatibility, pin to a 0.15.x release:

.. code-block:: bash

    pip install "skchange<0.16"

Installation
------------
The library can be installed via pip:

.. code-block:: bash

    pip install skchange

Requires python versions >= 3.10, < 3.15.

For better computational performance, it is recommended to install skchange with `numba <https://numba.readthedocs.io>`_:

.. code-block:: bash

    pip install skchange[numba]

Key features
------------

- **Fast**: `Numba <https://numba.readthedocs.io>`_ is used for performance.
- **Easy to use**: Follows the conventions of `scikit-learn <https://scikit-learn.org>`_.
- **Easy to extend**: Make your own detectors by inheriting from the base class templates. Create custom detection scores and cost functions.
- **Segment anomaly detection**: Detect intervals of anomalous behaviour in time series data.
- **Subset anomaly detection**: Detect intervals of anomalous behaviour in time series data, and infer the subset of variables that are responsible for the anomaly.

Mission
-------
The goal of ``skchange`` is to provide a library for fast and easy-to-use changepoint-based algorithms for change and anomaly detection.
The primary focus is on modern methods in the statistical literature.


Quick example
-------------

.. code-block:: python

    from skchange.new_api.datasets import generate_piecewise_normal_data
    from skchange.new_api.detectors import MovingWindow

    df = generate_piecewise_normal_data(
        means=[0, 5, 10, 5, 0], lengths=[50] * 5, seed=1,
    )
    cps = MovingWindow(bandwidth=20).fit(df).predict_changepoints(df)
    # array([ 50, 100, 150, 200])

See the :doc:`user_guide/index` for more, including multivariate anomaly
detection with variable identification, or jump to the
:doc:`api_reference/index`.

Licence
-------
This project is a free and open-source software licensed under the
`BSD 3-clause license <https://github.com/NorskRegnesentral/skchange/blob/main/LICENSE>`_.


.. toctree::
    :maxdepth: 2
    :hidden:

    user_guide/index
    api_reference/index
    developer_guide/index
    releases
