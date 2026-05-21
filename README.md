# skchange

[![codecov](https://codecov.io/gh/NorskRegnesentral/skchange/graph/badge.svg?token=QSS3AY45KY)](https://codecov.io/gh/NorskRegnesentral/skchange)
[![tests](https://github.com/NorskRegnesentral/skchange/actions/workflows/test.yml/badge.svg)](https://github.com/NorskRegnesentral/skchange/actions/workflows/test.yml)
[![docs](https://readthedocs.org/projects/skchange/badge/?version=latest)](https://skchange.readthedocs.io/en/latest/?badge=latest)
[![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/NorskRegnesentral/skchange/blob/main/LICENSE)
[![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python](https://img.shields.io/pypi/pyversions/skchange)](https://pypi.org/project/skchange/)
[![PyPI Downloads](https://static.pepy.tech/badge/skchange)](https://pepy.tech/projects/skchange)


<!-- [skchange]((https://skchange.readthedocs.io/en/latest/)) provides [scikit-learn](https://scikit-learn.org/)-like changepoint detection algorithms. -->

**Breaking changes expected.** skchange is undergoing a significant API redesign in upcoming releases.
See [Issue #120](https://github.com/NorskRegnesentral/skchange/issues/120) and the
[migration guide](https://github.com/NorskRegnesentral/skchange/blob/main/skchange/new_api/MIGRATION_GUIDE.md) for details.

- **New API (recommended)** is previewed in `skchange.new_api.*` and becomes the default in 0.17.0, when the same names move to top-level (`skchange.detectors`, `skchange.interval_scorers`, `skchange.penalties`, ...). Drop `new_api.` from imports when upgrading. Still experimental.
- **Current API** (`skchange.change_detectors`, `skchange.costs`, ...) emits a `FutureWarning` in 0.16.x and is removed in 0.17.0.

If you need stability and the old [sktime](https://www.sktime.net/) compatibility, pin to a 0.15.x release:
> ```sh
> pip install "skchange<0.16"
> ```


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

**New API** (recommended; default in 0.17.0):
```python
from skchange.new_api.datasets import generate_piecewise_normal_data
from skchange.new_api.detectors import MovingWindow

df = generate_piecewise_normal_data(
    means=[0, 5, 10, 5, 0],
    lengths=[50, 50, 50, 50, 50],
    seed=1,
)

detector = MovingWindow(bandwidth=20)
cps = detector.fit(df).predict_changepoints(df)
# array([ 50, 100, 150, 200])

labels = detector.predict(df)
# array([0, 0, ..., 1, 1, ..., 2, 2, ..., 3, 3, ..., 4, 4])
```

**Current API** (deprecated in 0.16.x, removed in 0.17.0):
```python
from skchange.change_detectors import MovingWindow
from skchange.datasets import generate_piecewise_normal_data

df = generate_piecewise_normal_data(
    means=[0, 5, 10, 5, 0],
    lengths=[50, 50, 50, 50, 50],
    seed=1,
)

detector = MovingWindow(bandwidth=20)
detector.fit_predict(df)
#    ilocs
# 0     50
# 1    100
# 2    150
# 3    200
```

### Multivariate anomaly detection with variable identification

**New API** (recommended; default in 0.17.0):
```python
from skchange.new_api.datasets import generate_piecewise_normal_data
from skchange.new_api.detectors import CAPA
from skchange.new_api.interval_scorers import L2Saving

df = generate_piecewise_normal_data(
    means=[0, 8, 0, 5],
    lengths=[100, 20, 130, 50],
    proportion_affected=[1.0, 0.1, 1.0, 0.5],
    n_variables=10,
    seed=1,
)

detector = CAPA(segment_saving=L2Saving())    # Looks for segments with non-zero means.
detector.fit(df)

anomalies = detector.predict_segment_anomalies(df)
# np.ndarray of shape (n_anomalies, 2); each row is [start, end)
# array([[100, 120],
#        [250, 300]])

# Full custom output per detector
# For CAPA: Intervals + affected feature indices + point anomalies):
result = detector.predict_all(df)
result["segment_anomalies"]         # same as predict_segment_anomalies above
result["segment_anomaly_features"]  # list of np.ndarray of affected feature indices, e.g.
# [array([0]),
#  array([2, 0, 4, 3, 1, 6, 8, 9, 7, 5])]
```

**Current API** (deprecated in 0.16.x, removed in 0.17.0):
```python
from skchange.anomaly_detectors import CAPA
from skchange.anomaly_scores import L2Saving
from skchange.compose.penalised_score import PenalisedScore
from skchange.datasets import generate_piecewise_normal_data
from skchange.penalties import make_linear_chi2_penalty

df = generate_piecewise_normal_data(
    means=[0, 8, 0, 5],
    lengths=[100, 20, 130, 50],
    proportion_affected=[1.0, 0.1, 1.0, 0.5],
    n_variables=10,
    seed=1,
)

score = L2Saving()  # Looks for segments with non-zero means.
penalty = make_linear_chi2_penalty(score.get_model_size(1), df.shape[0], df.shape[1])
penalised_score = PenalisedScore(score, penalty)
detector = CAPA(penalised_score, find_affected_components=True)
detector.fit_predict(df)
#         ilocs  labels         icolumns
# 0  [100, 120)       1              [0]
# 1  [250, 300)       2  [2, 0, 3, 1, 4]
```

## License

skchange is a free and open-source software licensed under the [BSD 3-clause license](https://github.com/NorskRegnesentral/skchange/blob/main/LICENSE).
