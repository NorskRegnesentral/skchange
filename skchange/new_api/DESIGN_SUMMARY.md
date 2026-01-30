# skchange New API Design Summary

Complete design documentation for the new changepoint detection API.

---

## Table of Contents
1. [Core Design Principles](#core-design-principles)
2. [Single-Series Design Decision](#single-series-design-decision)
3. [Output Type Design](#output-type-design)
4. [Sklearn Compatibility](#sklearn-compatibility)
5. [Meta-Estimator Pattern](#meta-estimator-pattern)
6. [Quick Reference](#quick-reference)

---

## Core Design Principles

### 1. Single-Series API
**Detectors operate on one time series at a time**

- **Input**: `detector.fit(X)` where `X` has shape `(n_samples, n_features)`
- **Output**: `predict(X)` returns single `Segmentation` dict
- **Univariate**: Always 2D with `n_features=1`, never 1D arrays
- **Multi-series workflows**: Handled externally via loops, GroupKFold, or parallel processing

**Why single-series only?**
- ✅ Full sklearn compatibility (pipelines, GridSearchCV, transformers)
- ✅ User controls memory and parallelization
- ✅ Simpler detector implementation
- ✅ Standard sklearn patterns for cross-series tuning (GroupKFold)
- ✅ No forced all-data-in-memory requirement

### 2. Sparse-First Representation
**Sparse changepoints as the canonical format**

Changepoint detection is fundamentally a **sparse problem** - most time series have few changepoints relative to their length.

**Core principle:**
- **Segmentation dict is the universal format** - Used for output, metrics, and conversions
- **Sparse by default** - `changepoints` array (length: n_changepoints) is primary
- **Dense on demand** - `transform()` or `sparse_to_dense()` converts to per-timepoint labels

**Why sparse-first?**
- ✅ **Natural representation** - `[50, 100, 150]` is clearer than 200-element array
- ✅ **Memory efficient** - O(k) vs O(n) where k << n
- ✅ **Algorithm-aligned** - Most detection algorithms work with changepoint locations
- ✅ **Metric-friendly** - Hausdorff, F1, etc. compare sparse locations directly
- ✅ **Self-contained** - `n_samples` enables standalone usage without X

### 3. TypedDict Output
**Plain dicts, sklearn-aligned**

- `predict()` returns `Segmentation` (TypedDict) directly
- Helper function: `make_segmentation()` for clean construction
- No custom classes - follows sklearn convention (GridSearchCV.cv_results_, cross_validate())
- Type hints without runtime overhead

### 4. Protocol-Based Architecture
**Duck typing by design**

- `ChangeDetector` is a `typing.Protocol`, not ABC
- **Required methods**: Only `fit()` and `predict()`
- **Optional methods**: `transform()`, `fit_transform()`, `fit_predict()`, `score()`
- **Inheritance is optional**: Any class with required methods works
- `BaseChangeDetector` provides convenience, not enforcement

### 5. Sklearn Alignment
**Follow scikit-learn conventions**

- **Naming**: `n_samples` (timepoints), `n_features` (variables/channels)
- **No custom classes**: Return plain dicts (TypedDict), not dataclass
- **Stateless predict()**: Returns values, doesn't modify state
- **Input flexibility**: Accept ArrayLike (Any), output np.ndarray
- **Unsupervised**: `y` parameter ignored (exists for API compatibility)

---

## Single-Series Design Decision

### The Decision

**Detectors accept only single time series in fit() and predict().**

Multi-series workflows are handled **outside** the detector using standard sklearn patterns.

### Rationale

**Problem with multi-series fit():**

1. **Breaks sklearn pipelines**:
   ```python
   # ❌ This fails with multi-series fit
   pipe = Pipeline([('scaler', StandardScaler()), ('detector', detector)])
   pipe.fit([X1, X2, X3])  # StandardScaler can't handle list
   ```

2. **Incompatible with GridSearchCV**:
   ```python
   X = [X1, X2, X3]  # list of series
   # ❌ GridSearchCV can't index: X[train_idx] fails
   grid.fit(X, y, groups=groups)
   ```

3. **Forces all data in memory**:
   ```python
   # ❌ Must load everything at once
   X_list = [load_series(i) for i in range(1000)]  # OOM!
   detector.fit(X_list)
   ```

4. **Removes user control**: User can't manage batching, streaming, or parallelization

**Solution - External multi-series handling:**

```python
# Hyperparameter tuning across series (GroupKFold)
X_all = np.vstack([X1, X2, X3])
groups = np.array([0]*len(X1) + [1]*len(X2) + [2]*len(X3))
cv = GroupKFold(n_splits=3)

grid = GridSearchCV(MyDetector(), param_grid, cv=cv)
grid.fit(X_all, y_all, groups=groups)  # ✅ Standard sklearn

# Parallel processing (joblib)
from joblib import Parallel, delayed

def process(X_i):
    detector.fit(X_i)
    return detector.predict(X_i)

results = Parallel(n_jobs=-1)(delayed(process)(X_i) for X_i in series_list)

# Memory-efficient streaming
for series_id in tqdm(all_series):
    X = load_series(series_id)
    detector.fit(X)
    result = detector.predict(X)
    save_result(result)
    del X  # Free memory
```

### Benefits of Single-Series Design

1. ✅ **Full sklearn compatibility**: Pipelines, GridSearchCV, cross_validate all work
2. ✅ **User controls resources**: Memory, parallelization, batching
3. ✅ **Simpler implementation**: No isinstance(X, list) checks, no dual code paths
4. ✅ **Standard patterns**: GroupKFold is well-documented sklearn approach
5. ✅ **Flexible workflows**: Streaming, distributed computing, progressive saving

### Multi-Series Use Cases - How to Handle

**Use case: Cross-series hyperparameter tuning**
```python
# Concatenate + GroupKFold
X_all = np.vstack(series_list)
groups = np.repeat(range(len(series_list)), [len(X) for X in series_list])
GridSearchCV(detector, params, cv=GroupKFold()).fit(X_all, groups=groups)
```

**Use case: Batch processing many series**
```python
# Simple loop with progress tracking
for X_i in tqdm(series_list):
    detector.fit(X_i)
    results.append(detector.predict(X_i))
```

**Use case: Learning shared parameters**
```python
# Custom meta-estimator wrapping base detector
class EnsembleDetector(BaseEstimator):
    def fit(self, X_list, y=None):
        self.detectors_ = [clone(self.base_detector).fit(X) for X in X_list]
        return self
```

### Rejected Alternatives

**❌ Option: Support both single and multi-series in fit()**
- Pros: Convenient for some algorithms
- Cons: Breaks pipelines, GridSearchCV, forces memory usage, adds complexity
- Decision: Convenience doesn't outweigh compatibility loss

**❌ Option: Separate classes (SingleSeriesDetector vs MultiSeriesDetector)**
- Pros: Clear separation
- Cons: API fragmentation, most algorithms don't need multi-series training
- Decision: External handling is more flexible

---

## Output Type Design

### Segmentation TypedDict

```python
from typing import TypedDict, NotRequired

class Segmentation(TypedDict):
    """Sparse segmentation representation."""
    # Required fields (must always be present)
    changepoints: np.ndarray         # Changepoint locations
    labels: np.ndarray               # Segment identifiers
    n_samples: int                   # Length of time series

    # Optional fields (NotRequired, can be omitted)
    n_features: NotRequired[int]                       # Number of features
    changed_features: NotRequired[list[np.ndarray]]    # Per-changepoint feature indices
    meta: NotRequired[dict[str, Any]]                  # Detector metadata
```

### Required Fields - Design Decisions

**Why `n_samples` is required:**
- ✅ **Critical for sparse-to-dense conversion** without keeping X:
  ```python
  result = detector.predict(X)
  # Later, X is gone
  dense = sparse_to_dense(result)  # Needs n_samples
  ```
- ✅ **Self-contained format**: Serialize/pass around independently
- ✅ **Validation**: Ensures changepoints are in valid range [0, n_samples)

**Why `labels` is required:**
- ✅ **Essential for dense conversion**: `sparse_to_dense()` needs segment assignments
- ✅ **Semantic information**: Not always [0,1,2,...] - some algorithms cluster segments
- ✅ **Safety**: Explicit over implicit (prevent silent bugs)
- ✅ **Auto-generated by helper**: `make_segmentation()` creates default labels

**Why `n_features` is optional:**
- Only useful for metadata/validation
- Can be inferred from X
- ⚠️ **IMPORTANT**: If `changed_features` is present, `n_features` should also be included
  - Cannot infer total features from changed subset (e.g., [0, 2] could mean 3, 4, or 10 features)
- Detectors can include in `meta` if needed

### Field Naming

**`changed_features` (not `affected_variables`):**
- Clearly indicates which features changed at each changepoint
- Distinguishes from segment labels (states vs changes)
- List of arrays: `[np.array([0, 2]), np.array([1])]` means first changepoint affects features 0,2
- ⚠️ **When included, also include `n_features`** (cannot infer total from subset)

### Why TypedDict?

**Sklearn philosophy:**
- ✅ No custom output classes (sklearn uses arrays or dicts)
- ✅ Stateless predict() - returns values, not attributes
- ✅ Zero coupling - third parties return dicts without importing
- ✅ Type hints without runtime overhead

**Examples from sklearn:**
- `GridSearchCV.cv_results_` returns dict
- `cross_validate()` returns dict
- `predict()` returns np.ndarray

### Alternatives Considered

| Approach | Sklearn-like | Type hints | Attribute access | Zero coupling |
|----------|--------------|------------|------------------|---------------|
| **TypedDict** ✅ | ✅ | ✅ | ❌ (dict keys) | ✅ |
| dataclass | ❌ | ✅ | ✅ | ❌ |
| NamedTuple | ⚠️ | ✅ | ✅ | ✅ |

**Rejected: dataclass**
- Custom class violates sklearn convention
- Tight coupling - users must import skchange
- Would be only sklearn-compatible library using dataclass outputs

**Rejected: NamedTuple**
- Immutability prevents updating `meta` dict
- Confusing tuple behavior (indexing)

### Helper Function

```python
def make_segmentation(
    changepoints: np.ndarray,
    n_samples: int,
    labels: np.ndarray | None = None,  # Auto-generated if None
    n_features: int | None = None,
    changed_features: list[np.ndarray] | None = None,
    meta: dict[str, Any] | None = None,
) -> Segmentation:
    """Create Segmentation with clean syntax, auto-generates labels."""
```

**Usage:**
```python
def predict(self, X):
    changepoints = self._detect(X)
    return make_segmentation(
        changepoints=changepoints,
        n_samples=len(X),
        n_features=X.shape[1],
        meta={"threshold": self.threshold_},
    )
```

### Removed Fields

**`scores` was removed:**
- Per-changepoint scores are artificial - most algorithms compute per-timepoint or per-interval statistics
- Natural home is separate method like `decision_function()` (similar to sklearn classifiers)
- Can be revisited when score semantics are better understood

---

## Sklearn Compatibility

### BaseChangeDetector Implementation

```python
class BaseChangeDetector(BaseEstimator):
    """Minimal base class for sklearn compatibility."""

    def __sklearn_tags__(self):
        """Custom tags for change detection."""
        return SkchangeTags(...)

    def transform(self, X):
        """Convert sparse to dense labels (convenience)."""
        return sparse_to_dense(self.predict(X))

    def fit_transform(self, X, y=None, **fit_params):
        """Sklearn compatibility - fit then transform."""
        return self.fit(X, y, **fit_params).transform(X)

    def fit_predict(self, X, y=None, **fit_params):
        """Sklearn compatibility - fit then predict."""
        return self.fit(X, y, **fit_params).predict(X)

    def score(self, X, y):
        """Sklearn compatibility - evaluate predictions."""
        from skchange.new_api.metrics import hausdorff_distance
        y_pred = self.predict(X)
        return -hausdorff_distance(y, y_pred)  # Higher is better
```

### The `y` Parameter

**Design: `y: ArrayLike | None = None` (ignored)**

**Rationale:**
- ✅ **Unsupervised**: Changepoint detection on single series is unsupervised
- ✅ **Sklearn compatibility**: Pipelines pass `y` through, must accept it
- ✅ **Flexible type**: ArrayLike allows future use without breaking changes
- ✅ **Documented**: "Ignored. Exists for sklearn API compatibility."

**Supervision happens externally:**
```python
# NOT via y parameter
detector.fit(X, y=changepoint_labels)  # ❌ y is ignored

# VIA cross-series tuning
GridSearchCV(detector, params, cv=GroupKFold()).fit(X_all, groups=series_ids)  # ✅
```

### Minimal Detector Implementation

```python
class MyDetector(BaseChangeDetector):
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = validate_data(self, X)  # Sets n_features_in_
        self.threshold_ = self.threshold * np.std(X)
        return self

    def predict(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        changepoints = self._detect_changepoints(X)
        return make_segmentation(
            changepoints=changepoints,
            n_samples=len(X),
            n_features=X.shape[1],
        )
```

**That's it!** BaseChangeDetector provides:
- `transform()` for dense labels
- `fit_transform()` and `fit_predict()` convenience methods
- `score()` with default metric
- `get_params()` / `set_params()` from BaseEstimator

---

## Meta-Estimator Pattern

### Composing Detectors with Other Estimators

Some algorithms use trained classifiers or other estimators as components.

**Pattern: Pass estimator as hyperparameter**

```python
class ClassifierChangeDetector(BaseChangeDetector):
    """Detect changes using a pre-trained classifier on sliding windows."""

    def __init__(self, classifier, window_size=10):
        self.classifier = classifier
        self.window_size = window_size

    def fit(self, X, y=None):
        from sklearn.utils.validation import check_is_fitted

        X = validate_data(self, X)

        # Require pre-fitted classifier
        try:
            check_is_fitted(self.classifier)
        except NotFittedError:
            raise ValueError(
                f"{self.__class__.__name__} requires a pre-fitted classifier. "
                "Train your classifier before passing it."
            )

        self.classifier_ = self.classifier
        return self

    def predict(self, X):
        # Apply classifier to sliding windows
        scores = [self.classifier_.predict(window) for window in windows(X)]
        changepoints = self._detect_from_scores(scores)
        return make_segmentation(changepoints, len(X))
```

**Usage:**
```python
# Train classifier separately
clf = RandomForestClassifier()
clf.fit(X_labeled, y_labeled)

# Use in detector
detector = ClassifierChangeDetector(classifier=clf, window_size=10)
detector.fit(X)
result = detector.predict(X_test)

# Works with GridSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100],  # Nested params!
    'window_size': [5, 10, 20],
}
grid = GridSearchCV(
    ClassifierChangeDetector(RandomForestClassifier()),
    param_grid
)
```

**This follows sklearn's meta-estimator pattern:**
- `CalibratedClassifierCV`, `BaggingClassifier`, `VotingClassifier` all do this
- Sub-estimator is a hyperparameter
- `check_is_fitted()` validates requirements
- Clear error messages guide users

---

## Quick Reference

### Minimal Implementation

```python
from skchange.new_api import BaseChangeDetector, make_segmentation
from skchange.new_api.utils import validate_data

class MyDetector(BaseChangeDetector):
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = validate_data(self, X)
        # Optional: learn from X
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        changepoints = self._detect(X)
        return make_segmentation(changepoints, len(X))
```

### Usage

```python
# Basic usage
detector = MyDetector(threshold=1.5)
detector.fit(X_train)
result = detector.predict(X_test)

print(result["changepoints"])  # np.array([50, 100])
print(result["labels"])        # np.array([0, 1, 2])

# Dense labels
labels = detector.transform(X_test)

# Cross-series tuning
X_all = np.vstack([X1, X2, X3])
groups = np.repeat(range(3), [len(X1), len(X2), len(X3)])

grid = GridSearchCV(
    MyDetector(),
    param_grid={"threshold": [0.5, 1.0, 2.0]},
    cv=GroupKFold(n_splits=3)
)
grid.fit(X_all, groups=groups)

# Parallel processing
from joblib import Parallel, delayed
results = Parallel(n_jobs=-1)(
    delayed(lambda X: MyDetector().fit(X).predict(X))(X_i)
    for X_i in series_list
)
```

---

## Summary

This design achieves:

✅ **Single-series API** - Full sklearn compatibility
✅ **Sparse-first representation** - Efficient, natural, self-contained
✅ **TypedDict output** - Plain dicts, zero coupling
✅ **Protocol-based** - Duck typing, inheritance optional
✅ **External multi-series** - User controls memory, parallelization
✅ **Meta-estimator support** - Compose with classifiers, regressors
✅ **Stateless predict()** - Thread-safe, reproducible
✅ **Minimal implementation** - ~20 lines per detector

**Key insights:**

1. **Single-series keeps it simple** - Multi-series workflows use standard sklearn patterns
2. **Sparse is canonical** - Dense available via `transform()` when needed
3. **TypedDict matches sklearn** - No custom output classes
4. **External > Internal** - GroupKFold, joblib give user more control than multi-series fit
5. **y is ignored** - Supervision via cross-series tuning, not labels

The design balances simplicity, flexibility, and sklearn alignment while respecting the realities of changepoint detection workflows.
