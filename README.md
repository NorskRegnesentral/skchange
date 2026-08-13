# skchange

[![codecov](https://codecov.io/gh/NorskRegnesentral/skchange/graph/badge.svg?token=QSS3AY45KY)](https://codecov.io/gh/NorskRegnesentral/skchange)
[![tests](https://github.com/NorskRegnesentral/skchange/actions/workflows/test.yml/badge.svg)](https://github.com/NorskRegnesentral/skchange/actions/workflows/test.yml)
[![docs](https://readthedocs.org/projects/skchange/badge/?version=latest)](https://skchange.readthedocs.io/en/latest/?badge=latest)
[![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/NorskRegnesentral/skchange/blob/main/LICENSE)
[![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python](https://img.shields.io/pypi/pyversions/skchange)](https://pypi.org/project/skchange/)
[![PyPI Downloads](https://static.pepy.tech/badge/skchange)](https://pepy.tech/projects/skchange)

Skchange provides fast and flexible changepoint detection algorithms within a [scikit-learn](https://scikit-learn.org/)-like API.
Users upgrading from 0.15.x should consult the [migration guide](MIGRATION_GUIDE.md).

## Documentation

* [Documentation](https://skchange.readthedocs.io/)
* [Notebook tutorial](https://github.com/sktime/sktime-tutorial-pydata-global-2024)


## Installation
It is recommended to install skchange with [numba](https://numba.readthedocs.io/en/stable/) for faster performance:
```sh
pip install skchange[numba]
```

Alternatively, you can install skchange without numba:
```sh
pip install skchange
```

## Quickstart

### Changepoint detection / time series segmentation

```python
from skchange.datasets import generate_piecewise_normal_data
from skchange.detectors import MovingWindow

X = generate_piecewise_normal_data(
    means=[0, 5, 10, 5, 0],
    lengths=[50, 50, 50, 50, 50],
    seed=1,
)

detector = MovingWindow(bandwidth=20)
detector.fit_predict(X)
```
```text
array([ 50, 100, 150, 200])
```

### Automatic penalty calibration

```python
import scipy.stats as st
from skchange.datasets import generate_piecewise_data
from skchange.detectors import SeededBinarySegmentation
from skchange.tuning import CalibratedDetector

# Change-free beta(2, 5) data used to calibrate the detection threshold.
X_calib = generate_piecewise_data(st.beta(2, 5), lengths=300, seed=0)

# Test data with two changepoints where the beta shape changes.
X = generate_piecewise_data(
    [st.beta(2, 5), st.beta(5, 2), st.beta(1, 10)],
    lengths=100,
    seed=1,
)

cal = CalibratedDetector(
    SeededBinarySegmentation(),
    level=0.05,
    n_simulations=999,
    random_state=0,
)
cal.fit(X_calib)
cal.predict(X)
```
```text
array([100, 200])
```

### Multivariate segment anomaly detection

```python
from skchange.datasets import generate_piecewise_normal_data
from skchange.detectors import CAPA
from skchange.interval_scorers import L2Saving

X = generate_piecewise_normal_data(
    means=[0, 8, 0, 5],
    lengths=[100, 20, 130, 50],
    proportion_affected=[1.0, 0.1, 1.0, 0.5],
    n_variables=10,
    seed=1,
)

detector = CAPA(segment_saving=L2Saving())
detector.fit(X)
detector.predict_segment_anomalies(X)
```
```text
array([[100, 120],
       [250, 300]])
```

## License

skchange is a free and open-source software licensed under the [BSD 3-clause license](https://github.com/NorskRegnesentral/skchange/blob/main/LICENSE).
