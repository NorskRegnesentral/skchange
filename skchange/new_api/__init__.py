"""New API design for handling single and multiple time series.

This module demonstrates the recommended pattern for building change detectors
that work on both single and multiple time series in a clean, maintainable way.

Design Philosophy
-----------------
1. **Flexible Inputs**: Accept ArrayLike | list[ArrayLike] (optimize for common case)
2. **Simple List Outputs**: Always return Segmentation dict (natural Python)
3. **Variable Lengths**: Use lists (not 3D arrays) for natural time series handling
4. **Protocol-Based**: Duck typing via typing.Protocol (inheritance optional)
5. **Base Class Dispatching**: Automatic routing based on input type + capability tags
6. **Minimal Implementation**: Detectors implement only what they need

Quick Start
-----------
```python
from skchange.new_api import BaseChangeDetector, Segmentation

class MyDetector(BaseChangeDetector):
    _tags = {"capability:multiple_series": True}

    def _fit(self, X, y=None):
        # X is guaranteed to be (n_timepoints, n_channels)
        self.threshold_ = compute_threshold(X)
        return self

    def _predict(self, X):
        changepoints = detect(X, self.threshold_)
        return make_segmentation(indices=changepoints, n_samples=len(X))

# Usage - works for single series
detector = MyDetector()
result = detector.predict(X)         # Segmentation dict
```

Documentation
-------------
- DESIGN_SUMMARY.md: Quick overview of design decisions
- README.md: Complete design documentation
- MIGRATION_GUIDE.md: How to implement detectors
- UNIFIED_OUTPUT_DESIGN.md: Why unified outputs are better
- examples.py: Three reference implementations
- examples_y_parameter.py: Flexible y parameter patterns
- test_pattern.py: Comprehensive test suite
```
"""

from skchange.new_api.base import BaseChangeDetector
from skchange.new_api.typing import (
    ArrayLike,
    ChangeDetector,
    Segmentation,
)
from skchange.new_api.utils import dense_to_sparse, make_segmentation, sparse_to_dense

__all__ = [
    "BaseChangeDetector",
    "ChangeDetector",
    "ArrayLike",
    "Segmentation",
    "make_segmentation",
    "sparse_to_dense",
    "dense_to_sparse",
]
