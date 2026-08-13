# Migration Guide: Old API → New API

This guide helps you migrate from the pandas-based API to the new sklearn-compatible API.

## Overview

skchange is transitioning from a pandas/sktime-based API to a numpy/sklearn-based API. Both APIs work on single series, but differ in:

## Timeline

**Note**: Since skchange is experimental (as stated in README), breaking changes are expected. The migration timeline is accelerated compared to stable libraries.

| Version | Status | Old API | New API |
|---------|--------|---------|------|
| **0.15.x** | Released | Default ✓ (silent) | Preview in `skchange` (feedback welcome) |
| **0.16.0** | Next release | Default, emits `FutureWarning` on import | Preview in `skchange` |
| **0.17.0** | Following release | **Removed** | **Default** ✓ at permanent locations; `skchange` raises `ImportError` pointing to the new locations |
| **0.18.0** | Cleanup | — | `skchange` stub removed |

**`skchange` is a temporary preview path** for early feedback. It will be replaced in 0.17.0 when the new API is promoted to its permanent location (e.g. `skchange.detectors`, `skchange.interval_scorers`, ...). Do not rely on the `skchange` import path in production code.

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

cps = detector.predict(df)  # pd.DataFrame with "ilocs" column
labels = detector.transform(df)  # pd.Series with segment labels
```

### New API
```python
from skchange.detectors import PELT
from skchange.datasets import generate_piecewise_normal_data

df = generate_piecewise_normal_data(means=[0, 5, 0], lengths=[50, 50, 50], seed=1)

detector = PELT(penalty=10.0)
detector.fit(df)  # ArrayLike input (pd.DataFrame, np.ndarray, ...)

cps = detector.predict(df)  # np.ndarray of changepoint indices
# For a per-sample dense-label view, post-process with the utility:
from skchange.utils.segmentation import changepoints_to_labels

labels = changepoints_to_labels(cps, n_samples=len(df))
```

**Key differences at a glance:**

| | Old API | New API |
|---|---|---|
| Input | `pd.DataFrame` | `ArrayLike`, 2D (`np.ndarray`, `pd.DataFrame`, ...) |
| Primary output (`predict`) | `pd.DataFrame` with `"ilocs"` column | `np.ndarray` of changepoint indices |
| Changepoints | `predict()["iloc"]` → `pd.Series` | `predict()` → `np.ndarray` |
| Dense labels | `transform()` → `pd.Series` | `changepoints_to_labels(cps, n_samples)` util → `np.ndarray` |
| Anomaly intervals | `predict()` → `pd.DataFrame` with `"ilocs"` column of `[start, end)` intervals | `predict_segment_anomalies()` → `np.ndarray` of shape `(n_anomalies, 2)` |
| Detector scores | Attribute (e.g. `scores_`) | `predict_scores()` → 1D `np.ndarray` (where supported) |
| Extras (cumulative costs, etc.) | Attributes | `predict_all()` → `dict` (where supported) |
| sklearn compatible | Limited | ✓ (pipelines, `clone`, `get_params`, `set_params`) |
| sktime compatible | ✓ | ✗ |
