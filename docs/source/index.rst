.. _home:

===================
Welcome to skchange
===================

A python library for fast change point and segment anomaly detection.
The library is designed to be compatible with `sktime <https://www.sktime.net>`_.
`Numba <https://numba.readthedocs.io>`_ is used for computational speed.

Installation
------------
The library can be installed via pip:

.. code-block:: bash

    pip install skchange

Requires python versions >= 3.10, < 3.14.

For better computational performance, it is recommended to install skchange with `numba <https://numba.readthedocs.io>`_:

.. code-block:: bash

    pip install skchange[numba]

Key features
------------

- **Fast**: `Numba <https://numba.readthedocs.io>`_ is used for performance.
- **Easy to use**: Follows the conventions of `sktime <https://www.sktime.net>`_ and `scikit-learn <https://scikit-learn.org>`_.
- **Easy to extend**: Make your own detectors by inheriting from the base class templates. Create custom detection scores and cost functions.
- **Segment anomaly detection**: Detect intervals of anomalous behaviour in time series data.
- **Subset anomaly detection**: Detect intervals of anomalous behaviour in time series data, and infer the subset of variables that are responsible for the anomaly.

Mission
-------
The goal of ``skchange`` is to provide a library for fast and easy-to-use changepoint-based algorithms for change and anomaly detection.
The primary focus is on modern methods in the statistical literature.


Example
-------
.. code-block:: python

    from skchange.anomaly_detectors import CAPA
    from skchange.anomaly_scores import L2Saving
    from skchange.compose.penalised_score import PenalisedScore
    from skchange.datasets import generate_anomalous_data
    from skchange.penalties import make_linear_chi2_penalty

    n = 300
    anomalies = [(100, 120), (250, 300)]
    means = [[8.0, 0.0, 0.0], [2.0, 3.0, 5.0]]
    df = generate_anomalous_data(n, anomalies, means, random_state=3)
    p = df.shape[1]

    score = L2Saving()
    penalty = make_linear_chi2_penalty(score.get_model_size(1), n, p)
    penalised_score = PenalisedScore(score, penalty)
    detector = CAPA(penalised_score, find_affected_components=True)
    detector.fit_predict(df)

.. code-block:: python

            ilocs  labels   icolumns
    0  [100, 120)       1        [0]
    1  [250, 300)       2  [2, 1, 0]

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
