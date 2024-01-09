# skchange
`skchange` provies sktime-compatible change detection and changepoint-based anomaly detection algorithms. Methods implement the annotator interface of sktime.

A playground for now.

[![codecov](https://codecov.io/gh/NorskRegnesentral/skchange/graph/badge.svg?token=QSS3AY45KY)](https://codecov.io/gh/NorskRegnesentral/skchange)
[![tests](https://github.com/NorskRegnesentral/skchange/actions/workflows/tests.yaml/badge.svg)](https://github.com/NorskRegnesentral/skchange/actions/workflows/tests.yaml)
[![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/sktime/sktime/blob/main/LICENSE)
[![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Quickstart
```python
from skchange.change_detectors.moscore import Moscore
from skchange.datasets.generate import generate_teeth_data

# Segment a time series
df = generate_teeth_data(n_segments=2, mean=10, segment_length=10, p=1, random_state=2)
detector = Moscore(bandwidth=5, fmt="dense")
detector.fit_predict(df)
>>>
0     0
1     0
2     0
3     0
4     0
5     0
6     0
7     0
8     0
9     0
10    1
11    1
12    1
13    1
14    1
15    1
16    1
17    1
18    1
19    1
Name: segment_id, dtype: int32

# Get the changepoints only (defined as the last index of a segment)
detector = Moscore(bandwidth=5, fmt="sparse")
detector.fit_predict(df)
>>>
9     0
19    1
Name: segment_id, dtype: int32
```

## Installation
```sh
pip install git+https://github.com/NorskRegnesentral/skchange
```

## Dependencies
- `pandas` >= 1.3
- `numpy` >= 1.19
- `numba` >= 0.56
- `sktime` >= 0.24

You also need Python >= 3.8.

Optional dependencies:
- Penalty tuning: `optuna` >= 3.1.1
- Plotting: `plotly` >= 5.13.0.


## License

`skchange` is a free and open-source software licensed under the [BSD 3-clause license](https://github.com/NorskRegnesentral/skchange/blob/main/LICENSE).
