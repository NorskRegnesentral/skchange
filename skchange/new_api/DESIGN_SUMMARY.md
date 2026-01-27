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

- `predict()` accepts only single series and returns `Segmentation` (TypedDict) directly
- Helper function: `make_segmentation()` for clean construction
- No type unions, no `@overload` decorators needed
- User code is simple: `result = detector.predict(X)` → access `result["changepoints"]`

**Benefits:**
- ✅ Single precise type for IDE autocomplete
- ✅ No awkward indexing: `result["changepoints"]` not `results[0]["changepoints"]`
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

### 5. Sparse-First Representation
**Sparse changepoints as the canonical format**

Changepoint detection is fundamentally a **sparse problem** - most time series have few changepoints relative to their length. The API embraces this with sparse-first design:

**Core principle:**
- **Segmentation dict is the universal format** - Used for output, input (y labels), metrics, and conversions
- **Sparse by default** - `changepoints` array (length: n_changepoints) is primary
- **Dense on demand** - `transform()` or `sparse_to_dense()` converts to per-timepoint labels

**Why sparse-first?**
- ✅ **Natural representation** - `[50, 100, 150]` is clearer than 200-element array
- ✅ **Memory efficient** - O(k) vs O(n) where k << n
- ✅ **Algorithm-aligned** - Most detection algorithms work with changepoint locations
- ✅ **Metric-friendly** - Hausdorff, F1, etc. compare sparse locations directly
- ✅ **Validation included** - `n_samples` enables consistency checks

**Format comparison:**
```python
# Sparse (canonical) - 3 changepoints in 200 samples
sparse = {
    "changepoints": np.array([50, 100, 150]),  # Just 3 values
    "labels": np.array([0, 1, 2, 3]),          # 4 segment IDs
    "n_samples": 200,
}

# Dense (convenience) - same information, 200 values
dense = np.array([0,0,0,...,1,1,1,...,2,2,2,...,3,3,3])  # 200 elements
```

**API usage:**
- `predict()` → Returns sparse Segmentation
- `transform()` → Returns dense labels (convenience wrapper)
- Metrics → Accept Segmentation (sparse), auto-convert if needed
- **y parameter** → **Only accepts Segmentation** for segment labels (strict sparse-first)

**Conversion utilities:**
- `sparse_to_dense(result)` → Convert Segmentation to per-timepoint labels
- `dense_to_sparse(labels, n_samples)` → Convert dense labels to Segmentation

---

## Output Type Design

### Decision: TypedDict over Dataclass

```python
from typing import TypedDict, NotRequired

class Segmentation(TypedDict):
    """Plain dict, sklearn-aligned output."""
    # Required (3 fields - must always be present)
    changepoints: np.ndarray         # Changepoint locations
    labels: np.ndarray               # Segment identifiers
    n_samples: int                   # Length of time series

    # Optional (4 fields - NotRequired, can be omitted)
    n_features: NotRequired[int]                       # Number of channels
    scores: NotRequired[np.ndarray]                    # Changepoint scores
    affected_variables: NotRequired[list[np.ndarray]]  # Per-CP variable indices
    meta: NotRequired[dict[str, Any]]                  # Detector metadata
```

**Field Usage Guidelines:**
- **Always include**: `changepoints`, `labels`, `n_samples` (required)
- **Include when relevant**: `n_features` (multivariate), `scores` (confidence), `affected_variables` (which channels changed)
- **Include for debugging**: `meta` (algorithm parameters, thresholds, timing)
- **Minimal valid result**: Just the 3 required fields

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
def make_segmentation(
    changepoints: np.ndarray,
    n_samples: int,
    labels: np.ndarray | None = None,  # Auto-generated if None
    n_features: int | None = None,  # Optional
    scores: np.ndarray | None = None,
    affected_variables: list[np.ndarray] | None = None,
    meta: dict[str, Any] | None = None,
) -> Segmentation:
    """Create result with clean syntax, auto-generates labels."""
    ...
```

**Usage in detectors:**
```python
def _predict(self, X: ArrayLike) -> Segmentation:
    cps = self._detect(X)
    return make_segmentation(
        changepoints=cps,
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

    def _predict(self, X: ArrayLike) -> Segmentation:
        changepoints = self._run_pelt(X)
        return make_segmentation(
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

    def _predict(self, X: ArrayLike) -> Segmentation:
        """Base class calls this for each series automatically."""
        changepoints = self._detect(X, self.threshold_)
        return make_segmentation(
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

    def _predict(self, X: ArrayLike) -> Segmentation:
        """Use shared parameters."""
        changepoints = run_pelt(X, self.global_threshold_)
        return make_segmentation(
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
    "changepoints": np.array([50, 100]),
    "labels": np.array([0, 1, 2]),
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
from skchange.new_api import BaseChangeDetector, make_segmentation
import numpy as np

class MyDetector(BaseChangeDetector):
    def _fit(self, X, y=None):
        # Learn from X (guaranteed 2D)
        self.threshold_ = np.std(X)
        return self

    def _predict(self, X):
        # Detect changepoints
        changepoints = np.array([len(X) // 2])
        return make_segmentation(
            indices=changepoints,
            n_samples=X.shape[0],
            n_features=X.shape[1],
        )
```

### Result Access

```python
result = detector.predict(X)

# Access fields (no indexing needed)
changepoints = result["changepoints"]
labels = result["labels"]
n = result["n_samples"]
scores = result.get("scores")  # Optional field

# For multiple series, use list comprehension
results = [detector.predict(X) for X in series_list]
for result in results:
    print(result["changepoints"])

# Helper auto-generates labels: [0, 1, 2, ...]
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

3. **Sparse-first representation**
   - Segmentation dict is the universal format (output, metrics, conversions)
   - Changepoint detection is inherently sparse: k changepoints in n samples where k << n
   - Dense labels available via transform() when needed
   - Memory efficient: O(k) vs O(n)

4. **TypedDict >> dataclass for sklearn alignment**
   - Plain dicts, no custom classes
   - Zero coupling, third-party friendly
   - Stateless predict() convention

5. **Helper function mitigates dict access UX**
   - Clean construction syntax
   - Auto-generates labels
   - Requires explicit n_samples (mirrors TypedDict required fields)

6. **Protocol-based >> inheritance-required**
   - Duck typing enables third-party integration
   - BaseChangeDetector is convenience, not requirement
   - Simpler for users, more flexible for ecosystem

---

## Files in This Directory

- **typing.py**: Protocol and TypedDict definitions
- **utils.py**: Helper functions (make_segmentation)
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
