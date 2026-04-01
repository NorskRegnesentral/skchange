# skchange New API Design Summary

Complete design documentation for the new changepoint detection API.

---

## Table of Contents
1. [Core Design Principles](#core-design-principles)
2. [Single-Series Design Decision](#single-series-design-decision)
3. [Output Type Design](#output-type-design)
4. [Sklearn Compatibility](#sklearn-compatibility)
5. [Meta-Estimator Pattern](#meta-estimator-pattern)
6. [Interval Scorer Design](#interval-scorer-design)
7. [Quick Reference](#quick-reference)

---

## Core Design Principles

### 1. Single-Series API
**Detectors operate on one (univariate or multivariate) time series at a time**

- **Input**: `detector.fit(X)` where `X` has shape `(n_samples, n_features)`
- **Output**: `predict(X)` returns `np.ndarray` of shape `(n_samples,)` — dense segment labels
- **Univariate**: Always 2D with `n_features=1`, never 1D arrays
- **Multi-series workflows**: Handled externally via loops or parallel processing

**Why single-series only?**
- ✅ Full sklearn compatibility (pipelines, cloning, metrics, get_params, set_params)
- ✅ User controls memory and parallelization
- ✅ Simpler detector implementation

### 2. Dual Output Format
**Dense labels from `predict()`, sparse changepoints from `predict_changepoints()`**

- `predict(X) -> np.ndarray (n_samples,)` — segment labels for each sample
mimics clustering output in sklearn, which is the closest task in sklearn: segment labels for each timepoint
- `predict_changepoints(X) -> np.ndarray (n_changepoints,)` — changepoint indices
- Subclasses implement `predict_changepoints()`; `predict()` is derived automatically via `changepoints_to_labels()` for most common detectors.
- Subclasses may override `predict()` directly when the algorithm natively produces dense labels

**Why `predict()` returns dense labels:**
- ✅ **Sklearn standard** — sklearn clusterers and classifiers return per-sample labels as arrays from `predict()`
- ✅ **Pipeline compatible** — downstream steps (scalers, classifiers) expect arrays
- ✅ **Metric compatible** — relevant sklearn scoring utilities work directly

**Why `predict_changepoints()`:**
- ✅ **Sparse representation** — `[50, 100, 150]` is clearer than a 200-element label array
- ✅ **Algorithm-aligned** — detection algorithms work with changepoint locations
- ✅ **Metric-friendly** — Hausdorff, F1, etc. compare changepoint locations directly

### 3. Additional Detector-Specific Output

Some detectors compute richer outputs alongside changepoints — test statistic scores, affected features, uncertainty estimates, etc. These are exposed through typed `predict_*` methods defined only on the concrete detector that computes them:

```python
def predict_scores(self, X) -> np.ndarray: # per-sample test statistic
def predict_proba(self, X) -> np.ndarray:  # posterior probability per sample
```

**Convention: `predict_all(X) -> dict`**

When a detector computes several outputs in a single pass, it may expose a `predict_all()` method returning all outputs as a dict. This is a **convenience method for power users** — not part of `BaseChangeDetector` and not a stable cross-detector contract:

```python
# Only on detectors that implement it
result = detector.fit(X).predict_all(X)
# Keys are detector-specific, e.g.:
# {"changepoints": np.ndarray, "scores": np.ndarray, "affected_features": list}
```

The typed `predict_*` methods remain the stable API. `predict_all()` is purely for convenience when the user wants everything in one call.

### 4. Sklearn Alignment (Where Possible)
**Follow scikit-learn conventions unless domain requirements prevent it**

- **Naming**: `n_samples` (timepoints), `n_features` (variables/channels)
- **No custom output classes**: Return `np.ndarray`
- **Stateless predict()**: Returns values, doesn't modify state
- **Fitted attributes**: Attributes set in `fit` end with `_` (e.g., `self.threshold_`).

**Where sklearn compatibility is intentionally broken:**
- `check_methods_sample_order_invariance` — time series is inherently order-sensitive
- `check_methods_subset_invariance` — detection requires ≥ 2 samples
- `GridSearchCV` / `cross_val_score` — concatenates series across folds, destroying series boundaries (see [Sklearn Compatibility](#sklearn-compatibility))

---

## Single-Series Design Decision

### The Decision

**Detectors accept only single time series in fit() and predict().**

Multi-series workflows are handled **outside** the detector using standard patterns (loops, joblib).

### Rationale

**Problem with multi-series fit():**

1. **Breaks sklearn pipelines**:
   ```python
   # ❌ This fails with multi-series fit
   pipe = Pipeline([('scaler', StandardScaler()), ('detector', detector)])
   pipe.fit([X1, X2, X3])  # StandardScaler can't handle list
   ```

2. **Forces all data in memory**:
   ```python
   # ❌ Must load everything at once
   X_list = [load_series(i) for i in range(1000)]  # OOM!
   detector.fit(X_list)
   ```

3. **Removes user control**: User can't manage batching, streaming, or parallelization

**Solution — External multi-series handling:**

```python
# Parallel processing (joblib)
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(lambda X: detector.fit(X).predict_changepoints(X))(X_i)
    for X_i in series_list
)

# Memory-efficient streaming
for series_id in tqdm(all_series):
    X = load_series(series_id)
    changepoints = detector.fit(X).predict_changepoints(X)
    save_result(changepoints)
    del X  # Free memory
```

### Benefits of Single-Series Design

1. ✅ **Simpler implementation**: No isinstance(X, list) checks, no dual code paths
2. ✅ **User controls resources**: Memory, parallelization, batching
3. ✅ **Standard patterns**: Loops, joblib, pipeline steps all work naturally
4. ✅ **Flexible workflows**: Streaming, distributed computing, progressive saving

### Rejected Alternatives

**❌ Option: Support both single and multi-series in fit()**
- Pros: Convenient for some algorithms
- Cons: Breaks pipelines, forces memory usage, adds complexity
- Decision: Convenience doesn't outweigh compatibility loss

**❌ Option: Separate classes (SingleSeriesDetector vs MultiSeriesDetector)**
- Pros: Clear separation
- Cons: API fragmentation, most algorithms don't need multi-series training
- Decision: External handling is more flexible

---

## Output Type Design

### Two Output Methods

```python
class BaseChangeDetector(BaseEstimator):
    def predict(self, X) -> np.ndarray:               # shape (n_samples,)
        """Dense segment labels — sklearn standard."""

    def predict_changepoints(self, X) -> np.ndarray:  # shape (n_changepoints,)
        """Sparse changepoint indices — subclasses implement this."""
```

**`predict()` is derived automatically:**
```python
def predict(self, X):
    changepoints = self.predict_changepoints(X)
    return changepoints_to_labels(changepoints, n_samples=len(X))
```

### `changepoints_to_labels()` Utility

Converts changepoint indices to a dense label array:

```python
from skchange.new_api.utils import changepoints_to_labels

changepoints = np.array([50, 100])
labels = changepoints_to_labels(changepoints, n_samples=150)
# labels: [0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ..., 2]
#          segment 0         segment 1         segment 2

# Optional custom labels
labels = changepoints_to_labels(changepoints, n_samples=150, labels=np.array([0, 1, 0]))
# Recurring pattern: segments 0 and 2 share label 0
```

### Design Philosophy

**Minimal implementations:** Subclasses implement only `predict_changepoints()`. `predict()` is derived for free. Subclasses may override `predict()` directly when the algorithm natively produces dense labels that might be shared across segments.

**Detector-specific metadata goes in fitted attributes:** Following sklearn convention, additional information (scores, thresholds, convergence info) is stored as fitted attributes (e.g., `detector.scores_`), not in the return value.

```python
def predict(self, X):
    changepoints, scores = self._detect(X)
    self.scores_ = scores          # Not in return value
    return changepoints            # Only the detection result
```


## Sklearn Compatibility

### What Works

```python
# Pipelines ✅
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('detector', MyDetector()),
])
pipe.fit(X_train)
labels = pipe.predict(X_test)   # np.ndarray (n_samples,)

# Cloning ✅
from sklearn.base import clone
detector_copy = clone(detector)

# get_params / set_params ✅ (via BaseEstimator inheritance)
detector.set_params(threshold=2.0)

# joblib-based parallelism ✅
from joblib import Parallel, delayed
results = Parallel(n_jobs=-1)(
    delayed(lambda X: detector.fit(X).predict(X))(X_i)
    for X_i in series_list
)
```

### What Doesn't Work — and Why

**GridSearchCV / cross_val_score ❌**

These tools require **sample order invariance** and treat all samples as i.i.d. observations from the same distribution. Change detection violates both:

1. **Series boundaries are ignored**: GroupKFold concatenates multiple test series into one array and passes it to `predict()`. The detector sees a single contiguous array that may span series from different underlying processes, producing meaningless results.

2. **Sample order is semantically meaningful**: Shuffling or subsetting rows of a time series destroys the structure that changepoint detection depends on.

```python
# ❌ This silently gives wrong results
cv = GroupKFold(n_splits=3)
scores = cross_val_score(detector, X_all, groups=groups, cv=cv)
# cross_val_score calls detector.predict(X_test_fold) where
# X_test_fold is a concatenation of multiple series — not a single series
```

**Sklearn estimator checks that fail (expected):**
- `check_methods_sample_order_invariance` — time series is order-sensitive by design
- `check_methods_subset_invariance` — detector requires ≥ 2 samples

### Cross-Series Hyperparameter Tuning

Since `GridSearchCV` doesn't work correctly, use a per-series loop instead:

```python
from sklearn.base import clone
from itertools import product

def multi_series_grid_search(detector, param_grid, series_list, scorer):
    """Manual grid search that respects series boundaries."""
    keys, values = zip(*param_grid.items())
    best_params, best_score = None, -np.inf

    for combo in product(*values):
        params = dict(zip(keys, combo))
        fold_scores = []
        for X_train, X_test, y_test in series_list:
            d = clone(detector).set_params(**params).fit(X_train)
            fold_scores.append(scorer(d, X_test, y_test))
        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score, best_params = mean_score, params

    return best_params
```

### Minimal Detector Implementation

```python
from sklearn.utils.validation import check_is_fitted, validate_data

class MyDetector(BaseChangeDetector):
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = validate_data(self, X)          # Sets n_features_in_
        self.threshold_ = self.threshold * np.std(X)
        return self

    def predict_changepoints(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self._detect(X)              # Returns np.ndarray of indices
```

`predict()` is automatically derived from `predict_changepoints()` by `BaseChangeDetector`.

### Parameter Validation

**Pattern: Use sklearn's parameter validation system**

```python
from skchange.new_api._utils_param_validation import (
    _fit_context,
    HasMethods,
    Interval,
    StrOptions,
)

class MyDetector(BaseChangeDetector):
    _parameter_constraints = {
        "threshold": [Interval(Real, 0, None, closed="neither")],
        "method": [StrOptions({"auto", "bottom_up", "max"})],
        "scorer": [HasMethods(["fit", "score"])],
        "min_length": [Interval(Integral, 1, None, closed="left")],
    }

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None):
        # Parameters automatically validated before this runs
        X = validate_data(self, X)
        return self
```

**Why `_utils_param_validation` module:**
- Centralizes imports from sklearn's private APIs
- Documents dependency on sklearn >= 1.6
- Single place to adapt if sklearn changes internal APIs

---

## Meta-Estimator Pattern

### Composing Detectors with Other Estimators

**Pattern: Pass estimator as hyperparameter**

```python
class ClassifierChangeDetector(BaseChangeDetector):
    def __init__(self, classifier, window_size=10):
        self.classifier = classifier
        self.window_size = window_size

    def fit(self, X, y=None):
        X = validate_data(self, X)
        check_is_fitted(self.classifier)    # Require pre-fitted classifier
        self.classifier_ = self.classifier
        return self

    def predict_changepoints(self, X):
        scores = [self.classifier_.predict(window) for window in windows(X)]
        return self._detect_from_scores(scores)
```

**This follows sklearn's meta-estimator pattern:**
- `CalibratedClassifierCV`, `BaggingClassifier`, `VotingClassifier` all do this
- Sub-estimator is a hyperparameter; `check_is_fitted()` validates it

---

## Interval Scorer Design

Interval scorers (costs, change scores, savings) provide the scoring functions used by change detection algorithms.

### Base Class vs Protocol

**Decision: Use `BaseIntervalScorer` base class, not Protocol**

**Rationale:**
- ✅ **Sklearn convention**: sklearn uses base classes (BaseEstimator), not Protocols
- ✅ **Heavy interface**: IntervalScorer has 7+ methods — too heavy for Protocol
- ✅ **NotImplementedError pattern**: Follows sklearn's pattern over ABC
- ✅ **Implementation sharing**: Base class provides common functionality (tags, validation)

**When to use Protocol vs Base Class:**
- **Protocol**: Minimal interface (2-3 methods), pure duck typing
- **Base Class**: Rich interface (5+ methods), shared implementation, inheritance beneficial

### NotImplementedError vs ABC

**Decision: Use `NotImplementedError` in base class methods, not `@abstractmethod`**

**Rationale:**
- ✅ **Sklearn pattern**: sklearn uses NotImplementedError, not ABC
- ✅ **Clearer errors**: Can provide custom error messages
- ✅ **No metaclass complexity**: Simpler class hierarchy

### Naming Conventions

**Rules:**
1. **Generic variable/parameter name**: Use `scorer` for all `BaseIntervalScorer` instances
2. **Base class name**: `BaseIntervalScorer` indicates the abstraction/concept
3. **Concrete class names**: Named after what they **produce**, not the abstraction

```python
# ✅ Concrete classes named after output
class L2Cost(BaseCost): ...
class CUSUM(BaseChangeScore): ...
class PenalisedScore(BaseIntervalScorer): ...

# ✅ Generic parameter uses "scorer"
class SomeDetector(BaseChangeDetector):
    def __init__(self, scorer, threshold=1.0):
        self.scorer = scorer
```

**Score vs Scorer terminology:**
- **scorer** (object): An instance of BaseIntervalScorer
- **score(s)** (values): Numeric values returned by `evaluate()`

### Type-Specific Base Classes

```
BaseIntervalScorer               # Root: fit(), evaluate(), interval_specs_ncols
├── BaseCost                    # Costs between intervals
│   ├── L2Cost
│   └── MultivariateGaussianCost
├── BaseChangeScore             # Scores for change detection
│   ├── CUSUM
│   └── CostChangeScore         # Adapter: cost → change score
├── BaseSaving                  # Global segment savings
└── BaseTransientScore          # Transient (epidemic) segment scores
```

---

## Quick Reference

### Minimal Implementation

```python
from skchange.new_api import BaseChangeDetector
from sklearn.utils.validation import check_is_fitted, validate_data

class MyDetector(BaseChangeDetector):
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = validate_data(self, X)
        self.threshold_ = self.threshold * np.std(X)
        return self

    def predict_changepoints(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self._detect(X)   # np.ndarray of changepoint indices
```

### Usage

```python
# Basic usage
detector = MyDetector(threshold=1.5)
detector.fit(X_train)

changepoints = detector.predict_changepoints(X_test)  # np.array([50, 100])
labels = detector.predict(X_test)                     # np.array([0,0,...,1,1,...,2])

# Pipelines
pipe = Pipeline([('scaler', StandardScaler()), ('detector', MyDetector())])
labels = pipe.fit(X_train).predict(X_test)

# Parallel processing across series
from joblib import Parallel, delayed
all_changepoints = Parallel(n_jobs=-1)(
    delayed(lambda X: MyDetector(threshold=1.5).fit(X).predict_changepoints(X))(X_i)
    for X_i in series_list
)

# Convert changepoints to dense labels manually
from skchange.new_api.utils import changepoints_to_labels
labels = changepoints_to_labels(changepoints, n_samples=len(X_test))
```

---
