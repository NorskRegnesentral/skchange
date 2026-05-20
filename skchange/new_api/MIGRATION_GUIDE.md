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
from skchange.change_detectors import PELT
from skchange.datasets import generate_piecewise_normal_data

df = generate_piecewise_normal_data(means=[0, 5, 0], lengths=[50, 50, 50], seed=1)

detector = PELT(penalty=10.0)
detector.fit(df)

cps = detector.predict(df)      # pd.DataFrame with "ilocs" column
labels = detector.transform(df) # pd.Series with segment labels
```

### New API
```python
from skchange.new_api.detectors import PELT
from skchange.datasets import generate_piecewise_normal_data

df = generate_piecewise_normal_data(means=[0, 5, 0], lengths=[50, 50, 50], seed=1)

detector = PELT(penalty=10.0)
detector.fit(df)  # ArrayLike input (pd.DataFrame, np.ndarray, ...)

labels = detector.predict(df)              # np.ndarray of per-sample segment labels
cps = detector.predict_changepoints(df)    # np.ndarray of changepoint indices
# Optional: detectors may expose `predict_all(X)` returning algorithm-specific
# extras as a dict (e.g. PELT's cumulative costs).
```

**Key differences at a glance:**

| | Old API | New API |
|---|---|---|
| Input | `pd.DataFrame` | `ArrayLike`, 2D (`np.ndarray`, `pd.DataFrame`, ...) |
| Primary output (`predict`) | `pd.DataFrame` with `"ilocs"` column | `np.ndarray` of per-sample segment labels |
| Changepoints | `predict()["iloc"]` → `pd.Series` | `predict_changepoints()` → `np.ndarray` |
| Dense labels | `transform()` → `pd.Series` | `predict()` → `np.ndarray` |
| Anomaly intervals | `predict()` → `pd.DataFrame` with `"ilocs"` column of `[start, end)` intervals | `predict_segment_anomalies()` → `np.ndarray` of shape `(n_anomalies, 2)` |
| Extras (cumulative costs, etc.) | Attributes | `predict_all()` → `dict` (where supported) |
| sklearn compatible | Limited | ✓ (pipelines, `clone`, `get_params`, `set_params`) |
| sktime compatible | ✓ | ✗ |
