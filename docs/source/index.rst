.. _home:

===================
Welcome to skchange
===================

A python library for fast collective anomaly and changepoint detection.
The library is designed to be compatible with `sktime <https://www.sktime.net>`_.
`Numba <https://numba.readthedocs.io>`_ is used for computational speed.

Installation
------------
The library can be installed via pip:

.. code-block:: bash

    pip install skchange

Requires python versions >= 3.9, < 3.13.

For better computational performance, it is recommended to install skchange with `numba <https://numba.readthedocs.io>`_:

.. code-block:: bash

    pip install skchange[numba]

Key features
------------

- **Fast**: `Numba <https://numba.readthedocs.io>`_ is used for performance.
- **Easy to use**: Follows the conventions of `sktime <https://www.sktime.net>`_ and `scikit-learn <https://scikit-learn.org>`_.
- **Easy to extend**: Make your own detectors by inheriting from the base class templates. Create custom detection scores and cost functions.
- **Collective anomaly detection**: Detect intervals of anomalous behaviour in time series data.
- **Subset collective anomaly detection**: Detect intervals of anomalous behaviour in time series data, and infer the subset of variables that are responsible for the anomaly.

Mission
-------
The goal of ``skchange`` is to provide a library for fast and easy-to-use changepoint-based algorithms for change and anomaly detection.
The primary focus is on modern methods in the statistical literature.


Example
-------
.. code-block:: python

    import numpy as np
    from skchange.anomaly_detectors import MVCAPA
    from skchange.datasets.generate import generate_anomalous_data

    n = 300
    anomalies = [(100, 120), (250, 300)]
    means = [[8.0, 0.0, 0.0], [2.0, 3.0, 5.0]]
    df = generate_anomalous_data(n, anomalies, means, random_state=3)

    detector = MVCAPA()
    detector.fit_predict(df)

.. code-block:: python

      anomaly_interval anomaly_columns
    0       [100, 120]             [0]
    1       [250, 300]       [2, 1, 0]

Licence
-------
This project is a free and open-source software licensed under the
`BSD 3-clause license <https://github.com/NorskRegnesentral/skchange/blob/main/LICENSE>`_.


.. toctree::
    :maxdepth: 2
    :hidden:

    user_guide
    api_reference
    developer_guide
    releases
