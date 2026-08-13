.. _home:

===================
Welcome to skchange
===================

Skchange provides fast and flexible changepoint detection algorithms within a
`scikit-learn <https://scikit-learn.org>`_-like API.
Users upgrading from 0.15.x should consult the
`migration guide <https://github.com/NorskRegnesentral/skchange/blob/main/MIGRATION_GUIDE.md>`_.

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

- **Theoretically grounded algorithms**: Fast exact and approximate search methods with solid statistical foundations.
- **High performance**: `Numba <https://numba.readthedocs.io>`_ is used extensively for computational speed.
- **Segment anomaly detection**: Detect intervals of anomalous behaviour in time series data.
- **High-dimensional data**: Algorithms covering settings where either few (sparse changes) or many features (dense changes) change simultaneously.
- **Automatic penalty calibration**: Data-driven utilities for calibrating the detection threshold to balance false alarms against missed detections.
- **Large scorer library**: A broad collection of built-in cost functions and statistical tests for a wide range of data distributions.
- **Easy to use**: Familiar ``fit`` / ``predict`` API for both users and contributors.
- **Easy to extend**: Inherit from base class templates to add custom costs and statistical tests for your dataset and problem.

Mission
-------
The goal of ``skchange`` is to provide a library for fast and easy-to-use offline changepoint detection algorithms.
We focus mainly on modern methods in the statistical literature.


Quick example
-------------

.. code-block:: python

    from skchange.datasets import generate_piecewise_normal_data
    from skchange.detectors import MovingWindow

    X = generate_piecewise_normal_data(
        means=[0, 5, 10, 5, 0], lengths=[50] * 5, seed=1,
    )
    cps = MovingWindow(bandwidth=20).fit_predict(X)
    # array([ 50, 100, 150, 200])

See the :doc:`user_guide/index` for more, or jump to the
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
