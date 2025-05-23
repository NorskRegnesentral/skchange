# skchange

[![codecov](https://codecov.io/gh/NorskRegnesentral/skchange/graph/badge.svg?token=QSS3AY45KY)](https://codecov.io/gh/NorskRegnesentral/skchange)
[![tests](https://github.com/NorskRegnesentral/skchange/actions/workflows/tests.yaml/badge.svg)](https://github.com/NorskRegnesentral/skchange/actions/workflows/tests.yaml)
[![docs](https://readthedocs.org/projects/skchange/badge/?version=latest)](https://skchange.readthedocs.io/en/latest/?badge=latest)
[![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/sktime/sktime/blob/main/LICENSE)
[![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python](https://img.shields.io/pypi/pyversions/skchange)](https://pypi.org/project/skchange/)
[![PyPI Downloads](https://static.pepy.tech/badge/skchange)](https://pepy.tech/projects/skchange)


[skchange]((https://skchange.readthedocs.io/en/latest/)) provides [sktime](https://www.sktime.net/)-compatible change detection and changepoint-based anomaly detection algorithms.

Experimental but maturing.

<!-- ## Latest release -->

Check out the latest version: [v0.13.0](https://github.com/NorskRegnesentral/skchange/releases)

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
from skchange.change_detectors import MovingWindow
from skchange.datasets import generate_alternating_data

df = generate_alternating_data(n_segments=10, segment_length=50, mean=5, random_state=1)

detector = MovingWindow(bandwidth=20)
detector.fit_predict(df)
```
```python
   ilocs
0     50
1    100
2    150
3    200
4    250
5    300
6    350
7    400
8    450
```

### Multivariate anomaly detection with variable identification
```python
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
```
```python
        ilocs  labels   icolumns
0  [100, 120)       1        [0]
1  [250, 300)       2  [2, 1, 0]
```

## License

skchange is a free and open-source software licensed under the [BSD 3-clause license](https://github.com/NorskRegnesentral/skchange/blob/main/LICENSE).
