# Migration Guide: Old API → New API

This guide helps you migrate from the pandas-based API to the new sklearn-compatible API.

## Overview

skchange is transitioning from a pandas/sktime-based API to a numpy/sklearn-based API. Both APIs work on single series, but differ in:

## Timeline

**Note**: Since skchange is experimental (as stated in README), breaking changes are expected. The migration timeline is accelerated compared to stable libraries.

| Version | Status | Old API | New API |
|---------|--------|---------|------|
| **0.14.3** | Current | Default ✓ | — |
| **0.15.0** | Next release | Works, no longer maintained | Preview in `skchange.new_api` (feedback welcome) |
| **0.16.0** | Following release | Removed | **Default** ✓ (stable, permanent location) |

**`skchange.new_api` is a temporary preview path** available in 0.15.0 for early feedback. It will be removed in 0.16.0 when the new API is promoted to its permanent location. Do not use `skchange.new_api` in production code.

**Pinning recommendation**: If you need stability right now, pin to the current version:
```bash
pip install "skchange==0.14.3"  # Stable old API, no breaking changes
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
