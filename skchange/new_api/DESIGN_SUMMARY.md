# skchange New API Design Summary

Complete design documentation for the new changepoint detection API.

---

## Table of Contents
1. [Core Design Principles](#core-design-principles)
2. [Output Type Design](#output-type-design)
3. [Implementation Patterns](#implementation-patterns)
4. [Naming Conventions](#naming-conventions)
5. [Architecture Overview](#architecture-overview)

---

## Core Design Principles

### 1. Input Flexibility
**Accept both single and multiple series naturally**

- **Single series**: `detector.fit(X)` where `X` has shape `(n_samples, n_features)`
- **Multiple series**: `detector.fit([X1, X2, X3])` - explicit list
- **Univariate**: Always 2D with `n_features=1`, never 1D arrays

**Why lists, not 3D arrays?**
- Time series have variable lengths (patient monitoring: 500h vs 3000h)
- Padding wastes memory and distorts algorithms
- Lists handle this naturally

### 2. Output Consistency
**Direct dict output from predict()**

- `predict()` accepts only single series and returns `ChangeDetectionResult` (TypedDict) directly
- Helper function: `make_change_detection_result()` for clean construction
- No type unions, no `@overload` decorators needed
- User code is simple: `result = detector.predict(X)` → access `result["indices"]`

**Benefits:**
- ✅ Single precise type for IDE autocomplete
- ✅ No awkward indexing: `result["indices"]` not `results[0]["indices"]`
- ✅ Simpler protocol (2 required methods: fit + predict)
- ✅ Better for pipelines and composition
- ✅ Easy batching: `[detector.predict(X) for X in series_list]`

### 3. Protocol-Based Architecture
**Duck typing by design**

- `ChangeDetector` is a `typing.Protocol`, not ABC
- **Required methods**: Only `fit()` and `predict()` (core domain logic)
- **Optional methods**:
  - `transform()`: Provided by BaseChangeDetector (sparse → dense conversion)
  - `get_params()` / `set_params()`: Provided by sklearn.base.BaseEstimator
- **Inheritance is optional**: Any class with required methods works
- `BaseChangeDetector` provides convenience (dispatching, validation, transform, sklearn compatibility), not enforcement
- Enables third-party detectors without modification or sklearn dependency

### 4. Sklearn Alignment
**Follow scikit-learn conventions**

- **Naming**: `n_samples` (timepoints), `n_features` (variables/channels)
- **No custom classes**: Return plain dicts (TypedDict), not dataclass
- **Stateless predict()**: Returns values, doesn't store attributes
- **Input flexibility, output consistency**: Accept ArrayLike, return np.ndarray

---

## Output Type Design

### Decision: TypedDict over Dataclass

```python
class ChangeDetectionResult(TypedDict, total=False):
    """Plain dict, sklearn-aligned output."""
    # Required
    indices: np.ndarray
    segment_labels: np.ndarray
    n_samples: int
    n_features: int
    # Optional
    scores: np.ndarray
    affected_variables: list[np.ndarray]
    meta: dict[str, Any]
```

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

| Approach | Sklearn-like | Type hints | Attribute access | Zero coupling | Defaults |
|----------|--------------|------------|------------------|---------------|----------|
| **TypedDict** ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| dataclass | ❌ | ✅ | ✅ | ❌ | ✅ |
| NamedTuple | ⚠️ | ✅ | ✅ | ✅ | ⚠️ |
| Bunch | ✅ | ❌ | ✅ | ❌ | ✅ |

**Rejected: dataclass**
- Custom class violates sklearn convention
- Tight coupling - users must import skchange
- Would be only sklearn-compatible library using dataclass outputs

**Rejected: NamedTuple**
- Immutability prevents updating `meta` dict
- Confusing tuple behavior (indexing)

**Rejected: Store as attributes**
- Violates stateless predict() convention
- Thread safety issues
- Batch prediction unclear

### Helper Function

To mitigate dict syntax, we provide a helper:

```python
def make_change_detection_result(
    indices: np.ndarray,
    n_samples: int,
    n_features: int,
    segment_labels: np.ndarray | None = None,  # Auto-generated if None
    scores: np.ndarray | None = None,
    affected_variables: list[np.ndarray] | None = None,
    meta: dict[str, Any] | None = None,
) -> ChangeDetectionResult:
    """Create result with clean syntax, auto-generates segment_labels."""
    ...
```

**Usage in detectors:**
```python
def _predict(self, X: ArrayLike) -> ChangeDetectionResult:
    changepoints = self._detect(X)
    return make_change_detection_result(
        indices=changepoints,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        meta={"threshold": self.threshold_},
    )
```

### Input vs Output Typing

**Inputs: Flexible (ArrayLike)**
- Accept numpy, pandas, lists
- `ArrayLike = Any` (avoid linter issues)

**Outputs: Consistent (np.ndarray)**
- Always numpy arrays
- Predictable for downstream processing
- Follows sklearn: flexible input, consistent output

---

## Implementation Patterns

Concrete detectors implement only what they need. Base class handles dispatching.

### Pattern 1: Single-Series Only

**Use when**: Algorithm fundamentally works on one series (e.g., PELT)

```python
class SimplePELT(BaseChangeDetector):
    _tags = {"capability:multiple_series": False}

    def _fit(self, X: ArrayLike, y=None) -> Self:
        """X guaranteed 2D: (n_samples, n_features)."""
        self.threshold_ = np.std(X) * self.penalty
        return self

    def _predict(self, X: ArrayLike) -> ChangeDetectionResult:
        changepoints = self._run_pelt(X)
        return make_change_detection_result(
            indices=changepoints,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            meta={"threshold": self.threshold_},
        )
```

**User experience:**
```python
detector = SimplePELT()
result = detector.predict(X_single)  # ✅ Works - returns dict directly
detector.fit([X1, X2])  # ❌ Clear error: doesn't support multiple series
```

### Pattern 2: Universal (Stateless)

**Use when**: No cross-series learning needed (e.g., Moving Window)

```python
class MovingWindowDetector(BaseChangeDetector):
    _tags = {"capability:multiple_series": True}

    def _fit(self, X: ArrayLike, y=None) -> Self:
        """Implement single-series logic only."""
        self.threshold_ = compute_threshold(X)
        return self

    def _predict(self, X: ArrayLike) -> ChangeDetectionResult:
        """Base class calls this for each series automatically."""
        changepoints = self._detect(X, self.threshold_)
        return make_change_detection_result(
            indices=changepoints,
            n_samples=X.shape[0],
            n_features=X.shape[1],
        )

    # That's it! Base class handles multiple series automatically
```

### Pattern 3: Batch-Optimized

**Use when**: Cross-series learning improves results (e.g., AutoML)

```python
class BatchPELT(BaseChangeDetector):
    _tags = {"capability:multiple_series": True}

    def _fit_multiple(self, X: list[ArrayLike], y=None) -> Self:
        """Learn shared parameters across series."""
        all_stds = [np.std(X_i) for X_i in X]
        global_std = np.median(all_stds)
        self.global_threshold_ = self._auto_tune(X) * global_std
        return self

    def _predict(self, X: ArrayLike) -> ChangeDetectionResult:
        """Use shared parameters."""
        changepoints = run_pelt(X, self.global_threshold_)
        return make_change_detection_result(
            indices=changepoints,
            n_samples=X.shape[0],
            n_features=X.shape[1],
        )
```

---

## Naming Conventions

### Sklearn Standard Names

Following `sklearn.base.BaseEstimator` conventions:

```python
# Standard sklearn shape
X.shape = (n_samples, n_features)

# Our results
result = {
    "indices": np.array([50, 100]),
    "segment_labels": np.array([0, 1, 2]),
    "n_samples": 200,      # sklearn: number of samples
    "n_features": 3,       # sklearn: number of features
}
```

**Naming mapping:**
- ~~`n_timepoints`~~ → **`n_samples`** (sklearn standard)
- ~~`n_channels`~~ → **`n_features`** (sklearn standard)

**Why sklearn naming?**
1. Consistency with sklearn ecosystem
2. Familiarity for sklearn users
3. Works seamlessly with sklearn tooling
4. Industry standard in ML/Python

---

## Architecture Overview

```
User API (Public)
    ↓
BaseChangeDetector (Dispatcher)
    • Check isinstance(X, list)
    • Validate capabilities via tags
    • Route to appropriate method
    ↓
┌─────────────────┬────────────────────┐
↓                 ↓                    ↓
_fit(X)       _fit_multiple(Xs)
_predict(X)   _predict_multiple(Xs)
    ↓                 ↓                    ↓
Concrete Detector Implementation
    • Single-only: just _fit/_predict
    • Universal: _fit/_predict (base handles batch)
    • Batch-opt: _fit_multiple + _predict
```

### Capability Tags

```python
_tags = {
    "capability:multiple_series": False,  # Single-series only
    "capability:multiple_series": True,   # Supports both
}
```

Base class validates and provides clear errors.

---

## Quick Reference

### Minimal Detector Implementation

```python
from skchange.new_api import BaseChangeDetector, make_change_detection_result
import numpy as np

class MyDetector(BaseChangeDetector):
    def _fit(self, X, y=None):
        # Learn from X (guaranteed 2D)
        self.threshold_ = np.std(X)
        return self

    def _predict(self, X):
        # Detect changepoints
        changepoints = np.array([len(X) // 2])
        return make_change_detection_result(
            indices=changepoints,
            n_samples=X.shape[0],
            n_features=X.shape[1],
        )
```

### Result Access

```python
result = detector.predict(X)

# Access fields (no indexing needed)
changepoints = result["indices"]
labels = result["segment_labels"]
n = result["n_samples"]
scores = result.get("scores")  # Optional field

# For multiple series, use list comprehension
results = [detector.predict(X) for X in series_list]
for result in results:
    print(result["indices"])

# Helper auto-generates segment_labels: [0, 1, 2, ...]
# Helper requires n_samples and n_features (explicit and clear)
```

---

## Key Design Insights

1. **Asymmetric API by design**
   - fit() accepts single or multiple series (shared parameter learning)
   - predict() accepts only single series (per-series operation)
   - Returns dict directly, not wrapped in list

2. **Lists >> 3D arrays for variable-length series**
   - Real time series have variable lengths
   - Padding wastes memory and distorts algorithms

3. **TypedDict >> dataclass for sklearn alignment**
   - Plain dicts, no custom classes
   - Zero coupling, third-party friendly
   - Stateless predict() convention

4. **Helper function mitigates dict access UX**
   - Clean construction syntax
   - Auto-generates segment_labels
   - Requires explicit n_samples/n_features (mirrors TypedDict)

5. **Protocol-based >> inheritance-required**
   - Duck typing enables third-party integration
   - BaseChangeDetector is convenience, not requirement
   - Simpler for users, more flexible for ecosystem

---

## Files in This Directory

- **typing.py**: Protocol and TypedDict definitions
- **utils.py**: Helper functions (make_change_detection_result)
- **base.py**: BaseChangeDetector with dispatching logic
- **examples.py**: Reference implementations (SimplePELT, MovingWindowDetector, BatchPELT)
- **examples_y_parameter.py**: Y parameter flexibility demos
- **DESIGN_SUMMARY.md**: This file

---

## Summary

This design achieves:

✅ **Minimal protocol** (2 required methods: fit, predict)
✅ **Optional transform()** (convenience method in BaseChangeDetector)
✅ **Sklearn alignment** (naming, no custom classes, stateless)
✅ **Predictable outputs** (always TypedDict)
✅ **Duck-typed interface** (Protocol, inheritance optional)
✅ **Natural inputs** (sklearn-compatible single series)
✅ **Variable lengths** (lists, not 3D arrays)
✅ **Minimal implementation** (~15 lines per detector)
✅ **Clear errors** (capability tags + validation)
✅ **Zero coupling** (plain dicts, third-party friendly)

The design balances simplicity, flexibility, and consistency while following sklearn best practices.
