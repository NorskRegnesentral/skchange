"""Example detector implementations showing the single/multi series pattern.

These examples demonstrate how concrete detectors implement only what they need.
"""

from __future__ import annotations

import numpy as np

from skchange.new_api.base import BaseChangeDetector
from skchange.new_api.typing import ArrayLike, Segmentation
from skchange.new_api.utils import make_segmentation

# ============================================================================
# Example 1: Single-Series Only Detector
# ============================================================================


class SimplePELT(BaseChangeDetector):
    """Example: Detector that only works on single series.

    This is the simplest case - just implement _fit and _predict.
    Users get a clear error if they try to pass multiple series.
    """

    def __init__(self, penalty: float = 1.0):
        self.penalty = penalty
        self.threshold_ = None

    def __sklearn_tags__(self):
        """Get estimator tags."""
        tags = super().__sklearn_tags__()
        tags.change_detector_tags.capability_multiple_series = False
        return tags

    def _fit(self, X: ArrayLike, y: ArrayLike | None = None) -> SimplePELT:
        """Fit on single series - core logic here.

        X is guaranteed to be 2D (n_samples, n_features).
        No need to handle list inputs - base class does that.
        """
        # Learn threshold from data
        self.threshold_ = np.std(X) * self.penalty
        return self

    def _predict(self, X: ArrayLike) -> Segmentation:
        """Detect changepoints in single series.

        X is guaranteed to be 2D (n_samples, n_features).
        """
        # Simplified PELT logic
        changepoints = self._run_pelt_algorithm(X)

        return make_segmentation(
            changepoints=changepoints,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            meta={"threshold": self.threshold_},
        )

    def _run_pelt_algorithm(self, X: ArrayLike) -> np.ndarray:
        """Run placeholder for actual PELT algorithm."""
        # Dummy implementation
        n = len(X)
        return np.array([n // 2]) if n > 10 else np.array([])


# ============================================================================
# Example 2: Universal Detector (Works on Both)
# ============================================================================


class MovingWindowDetector(BaseChangeDetector):
    """Example: Detector that works on both single and multiple series.

    Strategy: Implement _fit for single series (core algorithm).
    Base class automatically handles multiple series by calling _fit on each.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.threshold_ = None

    def __sklearn_tags__(self):
        """Get estimator tags."""
        tags = super().__sklearn_tags__()
        tags.capability_multiple_series = True
        return tags

    def _fit(self, X: ArrayLike, y: ArrayLike | None = None) -> MovingWindowDetector:
        """Fit on single series - this is all you need to implement.

        Base class will call this on each series if multiple are provided.
        """
        # Learn threshold from training data
        self.threshold_ = 2.0 * np.std(X)
        return self

    def _predict(self, X: ArrayLike) -> Segmentation:
        """Detect on single series."""
        n = len(X)
        scores = np.zeros(n)

        # Compute moving window scores
        for t in range(self.window_size, n - self.window_size):
            before = X[t - self.window_size : t]
            after = X[t : t + self.window_size]
            scores[t] = np.abs(np.mean(after) - np.mean(before))

        # Find peaks above threshold
        changepoints = np.where(scores > self.threshold_)[0]

        return make_segmentation(
            changepoints=changepoints,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            scores=scores[changepoints],  # Only scores at changepoints
            meta={"window_size": self.window_size},
        )

    # That's it! No need to implement _fit_multiple or _predict_multiple.
    # Base class handles it automatically.


# ============================================================================
# Example 3: Batch-Optimized Detector
# ============================================================================


class BatchPELT(BaseChangeDetector):
    """Example: Detector optimized for batch processing.

    This detector learns shared parameters across multiple series,
    so it implements custom _fit_multiple logic.
    """

    def __init__(self, penalty: float | None = None):
        self.penalty = penalty
        self.global_threshold_ = None

    def __sklearn_tags__(self):
        """Get estimator tags."""
        """Get estimator tags."""
        tags = super().__sklearn_tags__()
        tags.change_detector_tags.capability_multiple_series = True
        return tags

    def _fit(self, X: ArrayLike, y: ArrayLike | None = None) -> BatchPELT:
        """Single series fit - compute series-specific threshold."""
        # This gets called when user does: detector.fit(single_series)
        self.global_threshold_ = np.std(X) * (self.penalty if self.penalty else 1.0)
        return self

    def _fit_multiple(
        self, X: list[ArrayLike], y: list[ArrayLike] | None = None
    ) -> BatchPELT:
        """Multiple series fit - learn SHARED threshold across all series.

        This is the optimized batch path.
        """
        # Compute global threshold from all series
        all_stds = [np.std(X_i) for X_i in X]
        global_std = np.median(all_stds)

        # Auto-tune penalty if not provided
        if self.penalty is None:
            self.penalty = self._auto_tune_penalty(X)

        self.global_threshold_ = global_std * self.penalty
        return self

    def _predict(self, X: ArrayLike) -> Segmentation:
        """Predict on single series using learned threshold."""
        # Use the global threshold
        changepoints = self._detect_with_threshold(X, self.global_threshold_)

        return make_segmentation(
            changepoints=changepoints,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            meta={"threshold": self.global_threshold_},
        )

    def _auto_tune_penalty(self, X_list: list[ArrayLike]) -> float:
        """Auto-tune penalty using multiple series."""
        # Dummy implementation
        return 1.5

    def _detect_with_threshold(self, X: ArrayLike, threshold: float) -> np.ndarray:
        """PELT detection with given threshold."""
        # Dummy implementation
        n = len(X)
        return np.array([n // 2]) if n > 10 else np.array([])


# ============================================================================
# Usage Examples
# ============================================================================


def demo_single_series_only():
    """Demo: Single-series only detector."""
    print("=== Single-Series Only Detector ===")

    detector = SimplePELT(penalty=2.0)

    # Works fine
    X_single = np.random.randn(100, 3)  # (n_samples, n_features)
    detector.fit(X_single)
    results = detector.predict(X_single)
    # Result is always a list
    print(f"Single series: {len(results[0].indices)} changepoints detected")

    # This will raise a clear error
    try:
        X_multiple = [np.random.randn(100, 3), np.random.randn(150, 3)]
        detector.fit(X_multiple)
    except ValueError as e:
        print(f"Error (expected): {e}")


def demo_universal_detector():
    """Demo: Universal detector that works on both."""
    print("\n=== Universal Detector ===")

    detector = MovingWindowDetector(window_size=20)

    # Single series - returns list with 1 result
    X_single = np.random.randn(200, 2)
    detector.fit(X_single)
    results = detector.predict(X_single)
    print(f"Single series: {len(results[0]['indices'])} changepoints detected")

    # Multiple series - returns list with N results
    X_multiple = [
        np.random.randn(100, 2),
        np.random.randn(150, 2),
        np.random.randn(200, 2),
    ]
    detector.fit(X_multiple)
    results = detector.predict(X_multiple)
    print(f"Multiple series: {len(results)} series processed")
    for i, result in enumerate(results):
        print(f"  Series {i}: {len(result['indices'])} changepoints")


def demo_batch_optimized():
    """Demo: Batch-optimized detector."""
    print("\n=== Batch-Optimized Detector ===")

    detector = BatchPELT()  # Auto-tune penalty from data

    # Multiple series - uses shared threshold
    X_multiple = [
        np.random.randn(100, 1),
        np.random.randn(150, 1),
        np.random.randn(200, 1),
    ]
    detector.fit(X_multiple)
    print(f"Learned global threshold: {detector.global_threshold_:.3f}")

    results = detector.predict(X_multiple)
    print(f"Processed {len(results)} series with shared parameters")

    # Single series - also works, returns list
    X_single = np.random.randn(100, 1)
    detector.fit(X_single)
    results = detector.predict(X_single)
    print(f"Single series also works: {len(results[0].indices)} changepoints")


if __name__ == "__main__":
    demo_single_series_only()
    demo_universal_detector()
    demo_batch_optimized()
