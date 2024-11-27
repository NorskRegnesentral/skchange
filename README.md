# [skchange](https://skchange.readthedocs.io/en/latest/)

[![codecov](https://codecov.io/gh/NorskRegnesentral/skchange/graph/badge.svg?token=QSS3AY45KY)](https://codecov.io/gh/NorskRegnesentral/skchange)
[![tests](https://github.com/NorskRegnesentral/skchange/actions/workflows/tests.yaml/badge.svg)](https://github.com/NorskRegnesentral/skchange/actions/workflows/tests.yaml)
[![docs](https://readthedocs.org/projects/skchange/badge/?version=latest)](https://skchange.readthedocs.io/en/latest/?badge=latest)
[![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/sktime/sktime/blob/main/LICENSE)
[![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`skchange` provides sktime-compatible change detection and changepoint-based anomaly detection algorithms.

Experimental but maturing.

## [Documentation](https://skchange.readthedocs.io/en/latest/)
Now available.


## Installation
It is recommended to install skchange with numba for faster performance:
```sh
pip install skchange[numba]
```

Alternatively, you can install skchange without numba:
```sh
pip install skchange
```
Requires Python >= 3.9, < 3.13.


## Quickstart

### Changepoint detection / time series segmentation
```python
from skchange.change_detectors.moscore import MovingWindow
from skchange.datasets.generate import generate_alternating_data

df = generate_alternating_data(n_segments=10, segment_length=50, mean=5, random_state=1)

detector = MovingWindow(bandwidth=10)
detector.fit_predict(df)
```
```python
0     49
1     99
2    149
3    199
4    249
5    299
6    349
7    399
8    449
Name: changepoint, dtype: int64
```

### Multivariate anomaly detection
```python
import numpy as np
from skchange.anomaly_detectors import MVCAPA
from skchange.datasets.generate import generate_anomalous_data

n = 300
anomalies = [(100, 119), (250, 299)]
means = [[8.0, 0.0, 0.0], [2.0, 3.0, 5.0]]
df = generate_anomalous_data(n, anomalies, means, random_state=3)

detector = MVCAPA()
detector.fit_predict(df)
```
```python
  anomaly_interval anomaly_columns
0       [100, 119]             [0]
1       [250, 299]       [2, 1, 0]
```


<!-- Optional dependencies:
- Penalty tuning: `optuna` >= 3.1.1
- Plotting: `plotly` >= 5.13.0. -->


## License

`skchange` is a free and open-source software licensed under the [BSD 3-clause license](https://github.com/NorskRegnesentral/skchange/blob/main/LICENSE).
