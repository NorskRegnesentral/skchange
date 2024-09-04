# skchange
`skchange` provies sktime-compatible change detection and changepoint-based anomaly detection algorithms.

A playground for now.

[![codecov](https://codecov.io/gh/NorskRegnesentral/skchange/graph/badge.svg?token=QSS3AY45KY)](https://codecov.io/gh/NorskRegnesentral/skchange)
[![tests](https://github.com/NorskRegnesentral/skchange/actions/workflows/tests.yaml/badge.svg)](https://github.com/NorskRegnesentral/skchange/actions/workflows/tests.yaml)
[![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/sktime/sktime/blob/main/LICENSE)
[![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Installation
```sh
pip install skchange
```
Requires Python >= 3.9, < 3.13.

## Quickstart

### Changepoint detection / time series segmentation
```python
from skchange.change_detectors.moscore import Moscore
from skchange.datasets.generate import generate_teeth_data

df = generate_teeth_data(n_segments=10, segment_length=50, mean=5, random_state=1)
detector = Moscore(bandwidth=10)
detector.fit_predict(df)
>>>
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
from skchange.anomaly_detectors.mvcapa import Mvcapa
from skchange.datasets.generate import generate_teeth_data

df = generate_teeth_data(
    n_segments=5,
    segment_length=50,
    p=10,
    mean=10,
    affected_proportion=0.2,
    random_state=2,
)
detector = Mvcapa(collective_penalty="sparse")
detector.fit_predict(df)
>>>
  anomaly_interval anomaly_columns
0         [50, 99]          [0, 1]
1       [150, 199]          [0, 1]
```


<!-- Optional dependencies:
- Penalty tuning: `optuna` >= 3.1.1
- Plotting: `plotly` >= 5.13.0. -->


## License

`skchange` is a free and open-source software licensed under the [BSD 3-clause license](https://github.com/NorskRegnesentral/skchange/blob/main/LICENSE).
