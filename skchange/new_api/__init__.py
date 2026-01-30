"""New API design for changepoint detection and time series segmentation.

This module provides a sklearn-compatible API for changepoint detection that
outputs sparse segmentation results (changepoints + labels).

Design Philosophy
-----------------
1. **Single-Series API**: Detectors accept ArrayLike (single series), not lists
2. **Sparse-First Output**: Return Segmentation dict (changepoints + labels)
3. **Protocol-Based**: Duck typing via typing.Protocol (inheritance optional)
4. **Sklearn Compatible**: Works with pipelines, GridSearchCV, cross_validate
5. **Minimal Implementation**: Detectors implement only _fit() and _predict()

Quick Start
-----------
```python
from skchange.new_api import BaseChangeDetector, make_segmentation

class MyDetector(BaseChangeDetector):
    def _fit(self, X, y=None):
        # X is guaranteed to be 2D: (n_samples, n_features)
        self.threshold_ = compute_threshold(X)
        return self

    def _predict(self, X):
        changepoints = detect(X, self.threshold_)
        return make_segmentation(changepoints=changepoints, n_samples=len(X))

# Usage
detector = MyDetector()
result = detector.predict(X)  # Returns Segmentation dict
```

Multi-Series Workflows
----------------------
Use standard sklearn patterns for cross-series hyperparameter tuning:
- GroupKFold for cross-validation across series
- joblib.Parallel for parallel processing
- Manual loops for custom aggregation

See DESIGN_SUMMARY.md for complete examples.

Documentation
-------------
- DESIGN_SUMMARY.md: Complete design rationale and patterns
- template_detector.py: Copy-paste template for new detectors
- examples.py: Reference implementations
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
