# skchange
skchange provies sktime-compatible change detection and changepoint-based anomaly detection algorithms. Methods implement the annotator interface of sktime.

A playground for now.

[![codecov](https://codecov.io/gh/NorskRegnesentral/skchange/graph/badge.svg?token=QSS3AY45KY)](https://codecov.io/gh/NorskRegnesentral/skchange)
[![tests](https://github.com/NorskRegnesentral/skchange/actions/workflows/tests.yaml/badge.svg)](https://github.com/NorskRegnesentral/skchange/actions/workflows/tests.yaml)
[![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/sktime/sktime/blob/main/LICENSE)
[![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Quickstart
```python
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

Skchange is a free and open-source software licensed under the [BSD 3-clause license](https://github.com/NorskRegnesentral/skchange/blob/main/LICENSE).
