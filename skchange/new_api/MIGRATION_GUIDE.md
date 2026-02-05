# Migration Guide: Old API → New API

This guide helps you migrate from the pandas-based API to the new sklearn-compatible API.

## Overview

skchange is transitioning from a pandas/sktime-based API to a numpy/sklearn-based API. Both APIs work on single series, but differ in:

**Old API (pandas/sktime)**:
- pandas DataFrame input/output
- Dense output format (per-sample labels)
- sktime compatibility (tags, BaseObject)

**New API (numpy/sklearn)**:
- ✅ NumPy array input/output - better performance
- ✅ Sparse output format - more efficient representation
- ✅ Full sklearn compatibility - GridSearchCV, Pipeline, cross_validate
- ✅ Simpler base classes - just BaseEstimator
- ✅ Clone-friendly - works seamlessly with sklearn ecosystem

## Timeline

**Note**: Since skchange is experimental (as stated in README), breaking changes are expected. The migration timeline is accelerated compared to stable libraries.

| Version | Status | Old API | New API |
|---------|--------|---------|------|
| **0.3.x** | Current | Default ✓ | Available in `new_api` |
| **0.4.0** | Next release | ⚠️ Deprecated | **Default** ✓ |
| **0.5.0** | Following release | Removed | Only API |

**Experimental Status**: Users are expected to handle breaking changes when using experimental software. We recommend pinning to specific versions if you need stability:
```bash
pip install skchange==0.3.x  # Pin to old API
```

## Quick Comparison

### Old API (Current Default)

```python
from skchange.change_detectors import PELT
import pandas as pd

# Pandas DataFrames (single series)
df = pd.DataFrame({"value": data})

# Fit and predict
detector = PELT()
detector.fit(df)
result = detector.predict(df)  # Returns pandas DataFrame (sparse format)
labels = detector.transform(df)  # Returns dense labels (pandas Series)
```

### New API (Recommended)

```python
from skchange.new_api.detectors import PELT
import numpy as np

# NumPy arrays (2D required, single series)
X = data.reshape(-1, 1)  # shape: (n_samples, n_features)

# Fit and predict
detector = PELT()
detector.fit(X)
result = detector.predict(X)  # Returns Segmentation dict (sparse)

# Access sparse results
changepoints = result["changepoints"]  # Array of locations
n_changepoints = len(changepoints)

# Dense labels via transform
dense_labels = detector.transform(X)  # NumPy array
```

## Key Differences

### 1. Input Format

#### Old API: Pandas DataFrames
```python
# Old: pandas with index
df = pd.DataFrame(
    {"sensor1": x1, "sensor2": x2},
    index=pd.date_range("2020-01-01", periods=100)
)
detector.fit(df)
```

#### New API: NumPy Arrays (2D)
```python
# New: numpy arrays, always 2D
X = np.column_stack([x1, x2])  # shape: (100, 2)
detector.fit(X)

# Univariate: still 2D!
X = data.reshape(-1, 1)  # shape: (100, 1), NOT (100,)
```

**Migration tip**: Use `.values` or `.to_numpy()`
```python
# Convert pandas to numpy
X = df.to_numpy()  # or df.values
detector.fit(X)
```

### 2. Output Format

#### Old API: pandas DataFrames (Both Sparse and Dense)
```python
# Old: Sparse output from predict()
result = detector.predict(df)
# Output: DataFrame with intervals
#   ilocs        labels
#   [0, 50)      0
#   [50, 120)    1
#   [120, 200)   2

# Old: Dense output from transform()
labels = detector.transform(df)
# Output: Series with same index as input
#   2020-01-01    0
#   2020-01-02    0
#   ...
#   2020-01-03    1
```

#### New API: Dict (Sparse) or Array (Dense)
```python
# New: Sparse dict output from predict()
result = detector.predict(X)

# Access components
changepoints = result["changepoints"]  # e.g., [50, 120]
n_samples = result["n_samples"]        # e.g., 200
# labels may or may not be present (optional field)

# New: Dense numpy array from transform()
dense_labels = detector.transform(X)   # NumPy array: [0,0,...,1,1,...,2,2,...]
```

**Migration tip**: Convert to DataFrame if needed
```python
# Get dense labels with original index
dense_labels = detector.transform(X)
df_labels = pd.DataFrame(
    {"labels": dense_labels},
    index=original_df.index
)
```

### 3. Multi-Series Handling

**Both APIs**: Single series only

#### Old API: Loop manually or use sktime utilities
```python
# Old: Process each series separately
results = [detector.fit(df_i).predict(df_i) for df_i in [df1, df2, df3]]
```
```

#### New API: Same approach - loop or concatenate with groups
```python
# New: Option 1 - Loop over series (same as old API)
results = [detector.fit(X_i).predict(X_i) for X_i in [X1, X2, X3]]

# New: Option 2 - Concatenate and use GroupKFold (sklearn pattern)
from sklearn.model_selection import GroupKFold

X_concat = np.vstack([X1, X2, X3])
groups = np.array([0]*len(X1) + [1]*len(X2) + [2]*len(X3))

cv = GroupKFold(n_splits=3)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(detector, X_concat, cv=cv, groups=groups)
```

### 4. sklearn Integration

#### Old API: Limited sklearn Support
```python
# Old: Doesn't work well with sklearn
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(detector, param_grid)  # May have issues
```

#### New API: Full sklearn Compatibility
```python
# New: Works seamlessly
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline

# GridSearch
grid = GridSearchCV(
    PELT(),
    param_grid={'penalty': [0.1, 1.0, 10.0]},
    cv=5
)
grid.fit(X)

# Pipelines
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('detector', PELT())
])
pipe.fit(X)

# Cross-validation with scoring
from sklearn.model_selection import cross_val_score
scores = cross_val_score(detector, X, cv=5)
```

## Detailed Migration Examples

### Example 1: Basic Changepoint Detection

#### Before (Old API)
```python
from skchange.change_detectors import PELT
import pandas as pd

df = pd.DataFrame({"value": generate_data()})
detector = PELT(penalty=10.0, min_segment_length=5)
detector.fit(df)

# Get changepoints
changepoints_df = detector.predict(df)
print(changepoints_df)

# Get dense labels
labels = detector.transform(df)
```

#### After (New API)
```python
from skchange.new_api.detectors import PELT
import numpy as np

X = generate_data().reshape(-1, 1)  # Make 2D
detector = PELT(penalty=10.0, min_segment_length=5)
detector.fit(X)

# Get changepoints (sparse)
result = detector.predict(X)
changepoints = result["changepoints"]
print(f"Changepoints at: {changepoints}")
print(f"Number of segments: {len(result['labels'])}")

# Get dense labels if needed
labels = detector.transform(X)
```

### Example 2: Multivariate Detection

#### Before (Old API)
```python
import pandas as pd

df = pd.DataFrame({
    "sensor1": data1,
    "sensor2": data2,
    "sensor3": data3
})

detector = PELT()
detector.fit(df)
result = detector.transform(df)
```

#### After (New API)
```python
import numpy as np

X = np.column_stack([data1, data2, data3])  # shape: (n, 3)

detector = PELT()
detector.fit(X)

# Sparse output
result = detector.predict(X)
print(f"Changepoints: {result['changepoints']}")
print(f"Features analyzed: {result['n_features']}")

# Dense labels
labels = detector.transform(X)
```

### Example 3: Hyperparameter Tuning

#### Before (Old API)
```python
# Manual tuning
best_score = -np.inf
best_penalty = None

for penalty in [0.1, 1.0, 10.0]:
    detector = PELT(penalty=penalty)
    detector.fit(df_train)
    score = custom_score(detector, df_test)
    if score > best_score:
        best_score = score
        best_penalty = penalty
```

#### After (New API)
```python
from sklearn.model_selection import GridSearchCV

# Automatic tuning with sklearn
param_grid = {
    'penalty': [0.1, 1.0, 10.0],
    'min_segment_length': [2, 5, 10]
}

grid = GridSearchCV(
    PELT(),
    param_grid=param_grid,
    cv=5,
    # scoring=custom_scorer  # Use sklearn scoring
)
grid.fit(X_train)

best_detector = grid.best_estimator_
print(f"Best parameters: {grid.best_params_}")
```

### Example 4: Multiple Series

#### Before (Old API)
```python
# Process each series separately
detector = PELT()
results = [detector.fit_predict(df_i) for df_i in [df1, df2, df3]]
```

#### After (New API)
```python
# Option 1: Process each series
detector = PELT()
results = []
for X_i in [X1, X2, X3]:
    detector.fit(X_i)
    results.append(detector.predict(X_i))

# Option 2: Cross-validation with groups
from sklearn.model_selection import GroupKFold

X_all = np.vstack([X1, X2, X3])
groups = np.concatenate([
    np.full(len(X1), 0),
    np.full(len(X2), 1),
    np.full(len(X3), 2)
])

cv = GroupKFold(n_splits=3)
scores = cross_val_score(
    detector, X_all,
    cv=cv, groups=groups
)
```

### Example 5: Anomaly Detection with Change Detector

#### Before (Old API)
```python
from skchange.anomaly_detectors import StatThresholdAnomaliser
from skchange.change_detectors import MovingWindow

change_det = MovingWindow(window_size=50)
anomaly_det = StatThresholdAnomaliser(
    change_detector=change_det,
    stat_lower=-2.0,
    stat_upper=2.0
)
anomaly_det.fit(df)
anomalies = anomaly_det.predict(df)
```

#### After (New API)
```python
# Pattern: Prefit or unfitted classifier/detector as parameter
from skchange.new_api.detectors import MovingWindow
from skchange.new_api.anomaly import StatThresholdAnomaliser  # When implemented

# Option 1: Pass unfitted detector (will be cloned and fitted)
change_det = MovingWindow(window_size=50)
anomaly_det = StatThresholdAnomaliser(
    change_detector=change_det,
    stat_lower=-2.0,
    stat_upper=2.0
)
anomaly_det.fit(X)
result = anomaly_det.predict(X)

# Option 2: Pass fitted detector (transfer learning)
change_det_fitted = MovingWindow(window_size=50).fit(X_train)
anomaly_det = StatThresholdAnomaliser(
    change_detector=change_det_fitted  # Already fitted
)
anomaly_det.fit(X)  # Uses pretrained change_det
result = anomaly_det.predict(X)
```

## Common Pitfalls

### 1. ❌ 1D Arrays
```python
# WRONG: 1D array
X = np.array([1, 2, 3, 4, 5])
detector.fit(X)  # Error!

# CORRECT: 2D array
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
detector.fit(X)  # Works
```

### 2. ❌ Expecting DataFrame Output
```python
# WRONG: Assuming pandas
result = detector.predict(X)
result.iloc[0]  # AttributeError: dict has no attribute 'iloc'

# CORRECT: Use dict access
changepoints = result["changepoints"]
labels = result["labels"]

# Or get dense labels
dense = detector.transform(X)  # NumPy array
```

### 3. ❌ Multi-Series Direct Fitting
```python
# WRONG: List of arrays
detector.fit([X1, X2, X3])  # Not supported!

# CORRECT: Loop or concatenate
for X_i in [X1, X2, X3]:
    detector.fit(X_i).predict(X_i)

# Or use GroupKFold
X_all = np.vstack([X1, X2, X3])
groups = ...
```

### 4. ❌ Modifying Parameters After Init
```python
# WRONG: Setting attributes directly
detector = PELT()
detector.penalty = 10.0  # This won't work as expected

# CORRECT: Use set_params or create new instance
detector.set_params(penalty=10.0)
# Or
detector = PELT(penalty=10.0)
```

## Benefits of Migration

### 1. sklearn Ecosystem
```python
# Now works seamlessly
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Complex pipelines
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('detector', PELT(penalty=1.0))
])

# Full CV with multiple metrics
cv_results = cross_validate(
    pipe, X,
    cv=5,
    scoring=['accuracy', 'precision'],
    return_train_score=True
)
```

### 2. Better Performance
```python
# Numpy operations, no pandas overhead
# Faster for large datasets
X = np.random.randn(100000, 10)
detector.fit(X)  # Faster than DataFrame version
```

### 3. Cleaner Code
```python
# Sparse output is more intuitive
result = detector.predict(X)
print(f"Found {len(result['changepoints'])} changepoints")
print(f"At locations: {result['changepoints']}")

# vs old dense output that needs processing
```

### 4. Composability
```python
# Easy to build custom workflows
from sklearn.base import clone

# Clone detector for different configs
base = PELT()
detectors = [
    clone(base).set_params(penalty=p)
    for p in [0.1, 1.0, 10.0]
]

# Ensemble different methods
results = [det.fit(X).predict(X) for det in detectors]
```

## FAQ

### Q: When should I migrate?
**A:** Start migrating immediately. The new API will become default in version 0.4.0 (next release) and the old API will be removed in 0.5.0. Since skchange is experimental, users are expected to handle breaking changes.

### Q: Can I use both APIs simultaneously?
**A:** Yes! During the transition period, both APIs are available:
```python
from skchange import PELT as OldPELT  # Old API
from skchange.new_api.detectors import PELT as NewPELT  # New API
```

### Q: What if I need pandas DataFrames?
**A:** Convert before/after:
```python
# Before detection
X = df.to_numpy()
result = detector.fit(X).predict(X)

# After detection
labels_df = pd.DataFrame(
    {"labels": detector.transform(X)},
    index=df.index
)
```

### Q: How do I handle datetime indices?
**A:** Store separately and reapply:
```python
# Store index
original_index = df.index

# Convert to numpy
X = df.to_numpy()

# Detect
result = detector.fit(X).predict(X)

# Map changepoints back to datetime
changepoint_times = original_index[result["changepoints"]]
```

### Q: Are there any algorithms not yet migrated?
**A:** As of version 0.3.x, the new API is being developed. Core algorithms like PELT, MovingWindow, and SeededBinarySegmentation will be prioritized for version 0.4.0. Since the library is experimental, we may not migrate all detectors - focus is on the most important ones. Check the documentation for current status.

### Q: Can I still use the old API after 0.5.0?
**A:** No, it will be removed. Since skchange is experimental, we don't maintain legacy APIs. Pin to version 0.3.x if you need the old API:
```bash
pip install skchange==0.3.*
```

### Q: What about custom detectors?
**A:** The new API is simpler to extend:
```python
from skchange.new_api.base import BaseChangeDetector

class MyDetector(BaseChangeDetector):
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        # Your logic
        return self

    def predict(self, X):
        # Return Segmentation dict
        return {
            "changepoints": ...,
            "labels": ...,
            "n_samples": len(X)
        }
```

## Getting Help

- **Documentation**: https://skchange.readthedocs.io/
- **New API Examples**: See `skchange/new_api/examples.py`
- **Template**: Use `skchange/new_api/template_detector.py` for custom detectors
- **Issues**: Report migration problems at https://github.com/NorskRegnesentral/skchange/issues

## Summary

| Feature | Old API | New API |
|---------|---------|---------|
| Input | pandas DataFrame | NumPy array (2D) |
| Output | Dense DataFrame | Sparse dict + optional dense |
| Multi-series | Loop manually | Loop or sklearn patterns |
| sklearn compat | Limited | Full |
| Performance | Slower (pandas) | Faster (numpy) |
| Complexity | Higher | Lower |
| Status | ⚠️ Deprecated | ✅ Recommended |

**Start your migration today to benefit from sklearn's ecosystem and cleaner API design!**
