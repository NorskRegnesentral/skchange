# Migration Guide: Old API → New API

This guide helps you migrate from the pandas-based API to the new sklearn-compatible API.

## Overview

skchange is transitioning from a pandas/sktime-based API to a numpy/sklearn-based API. Both APIs work on single series, but differ in:

## Timeline

**Note**: Since skchange is experimental (as stated in README), breaking changes are expected. The migration timeline is accelerated compared to stable libraries.

| Version | Status | Old API | New API |
|---------|--------|---------|------|
| **0.15.x** | Released | Default ✓ (silent) | Preview in `skchange.new_api` (feedback welcome) |
| **0.16.0** | Next release | Default, emits `FutureWarning` on import | Preview in `skchange.new_api` |
| **0.17.0** | Following release | **Removed** | **Default** ✓ at permanent locations; `skchange.new_api` raises `ImportError` pointing to the new locations |
| **0.18.0** | Cleanup | — | `skchange.new_api` stub removed |

**`skchange.new_api` is a temporary preview path** for early feedback. It will be replaced in 0.17.0 when the new API is promoted to its permanent location (e.g. `skchange.detectors`, `skchange.interval_scorers`, ...). Do not rely on the `skchange.new_api` import path in production code.

**Pinning recommendation**: If you need stability right now, pin to a 0.15.x release:
```bash
pip install "skchange<0.16"  # Old API, no deprecation warnings
pip install "skchange<0.17"  # Old API still works (warnings in 0.16.x)
```

To surface old-API usage in your test suite during the 0.16.x cycle:
```bash
pytest -W error::FutureWarning
```

## Quick Comparison

### Old API
```python
import pandas as pd
from skchange.change_detectors import PELT
from skchange.datasets import generate_piecewise_normal_data

df = generate_piecewise_normal_data(means=[0, 5, 0], lengths=[50, 50, 50], seed=1)

detector = PELT(penalty=10.0)
detector.fit(df)

changepoints = detector.predict(df)  # Returns pd.DataFrame with "ilocs" column
labels = detector.transform(df)      # Returns pd.Series with segment labels
```

### New API
```python
from skchange.new_api.detectors import PELT
from skchange.datasets import generate_piecewise_normal_data

df = generate_piecewise_normal_data(means=[0, 5, 0], lengths=[50, 50, 50], seed=1)

detector = PELT(penalty=10.0)
detector.fit(df)  # ArrayLike input supported (pd.DataFrame, np.ndarray, etc.)

result = detector.predict(df)         # Returns Segmentation dict
changepoints = result["changepoints"]  # np.ndarray of changepoint locations
labels = detector.transform(df)       # Returns np.ndarray of segment labels
```

**Key differences at a glance:**

| | Old API | New API |
|---|---|---|
| Input | `pd.DataFrame` | `np.ndarray` (2D) |
| `predict()` output | `pd.DataFrame` with `"ilocs"` column | `dict` with `"changepoints"` key |
| `transform()` output | `pd.Series` | `np.ndarray` |
| sklearn compatible | Limited | ✓ |
| sktime compatible | ✓ | ✗ |
