.. _home:

===================
Welcome to skchange
===================

Skchange provides fast and flexible changepoint detection algorithms within a
`scikit-learn <https://scikit-learn.org>`_-like API.
Users upgrading from version <0.17 should consult the
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

- **Theoretical soundness**: Exact and approximate changepoint detection algorithms with solid statistical foundations.
- **Flexible**: Detectors are composed of modular costs or statistical tests. Browse the :doc:`api_reference/interval_scorers` for built-in options or see :doc:`developer_guide/extending` to implement your own.
- **Fast**: `Numba <https://numba.readthedocs.io>`_ is used extensively for computational speed.
- **Easy to use**: Familiar `scikit-learn <https://scikit-learn.org>`_ ``fit`` / ``predict`` API for both users and contributors.
- **Segment anomaly detection**: Detect intervals of anomalous behaviour in time series data.
- **High-dimensional data**: Algorithms suitable for high-dimensional data with an unknown number of changing features.
- **Automatic penalty calibration**: Data-driven utilities for calibrating the false alarm rate.

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

See the :doc:`user_guide/index` for more, or jump to the :doc:`api_reference/index`.

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
