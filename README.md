# skchange
`skchange` provies sktime-compatible change detection and changepoint-based anomaly detection algorithms. Methods implement the annotator interface of sktime.

A playground for now.

[![codecov](https://codecov.io/gh/NorskRegnesentral/skchange/graph/badge.svg?token=QSS3AY45KY)](https://codecov.io/gh/NorskRegnesentral/skchange)
[![tests](https://github.com/NorskRegnesentral/skchange/actions/workflows/tests.yaml/badge.svg)](https://github.com/NorskRegnesentral/skchange/actions/workflows/tests.yaml)
[![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/sktime/sktime/blob/main/LICENSE)
[![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Quickstart
```python
from skchange.change_detectors.mosum import Mosum
from skchange.datasets.generate import teeth

# Predict the segment membership of each observation
df = teeth(n_segments=2, mean=10, segment_length=10, p=1, random_state=2)
detector = Mosum(bandwidth=5, fmt="dense")
detector.fit_predict(df)
>>>
0     0.0
1     0.0
2     0.0
3     0.0
4     0.0
5     0.0
6     0.0
7     0.0
8     0.0
9     0.0
10    1.0
11    1.0
12    1.0
13    1.0
14    1.0
15    1.0
16    1.0
17    1.0
18    1.0
19    1.0
dtype: float64

# Predict the changepoints (the last index of each segment)
detector = Mosum(bandwidth=5, fmt="sparse")
detector.fit_predict(df)
>>>
9     0.0
19    1.0
dtype: float64
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
