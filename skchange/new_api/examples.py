"""Example detector implementations showing single-series changepoint detection.

These examples demonstrate the minimal implementation required for detectors.
All detectors work on single series only (ArrayLike input, not lists).
"""

from __future__ import annotations

import numpy as np

from skchange.new_api.base import BaseChangeDetector
from skchange.new_api.typing import ArrayLike, Segmentation
from skchange.new_api.utils import make_segmentation

# ============================================================================
# Example 1: Simple Detector
# ============================================================================


class SimplePELT(BaseChangeDetector):
    """Example: Simple PELT-like changepoint detector.

    Demonstrates minimal implementation - just fit() and predict().
    """

    def __init__(self, penalty: float = 1.0):
        self.penalty = penalty
        self.threshold_ = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> SimplePELT:
        """Fit on single series.

        X is guaranteed to be 2D (n_samples, n_features).
        """
        # Learn threshold from data
        self.threshold_ = np.std(X) * self.penalty
        return self

    def predict(self, X: ArrayLike) -> Segmentation:
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
# Example 2: Moving Window Detector
# ============================================================================


class MovingWindowDetector(BaseChangeDetector):
    """Example: Moving window-based changepoint detector.

    Demonstrates stateful fitting (learns threshold from training data).
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.threshold_ = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> MovingWindowDetector:
        """Fit on single series."""
        # Learn threshold from training data
        self.threshold_ = 2.0 * np.std(X)
        return self

    def predict(self, X: ArrayLike) -> Segmentation:
        """Detect changepoints using moving window."""
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
            meta={"window_size": self.window_size},
        )


# ============================================================================
# Usage Examples
# ============================================================================


def demo_simple_detector():
    """Demo: Simple PELT detector."""
    print("=== Simple PELT Detector ===")

    detector = SimplePELT(penalty=2.0)

    # Fit and predict on single series
    X = np.random.randn(100, 3)  # (n_samples, n_features)
    detector.fit(X)
    result = detector.predict(X)

    print(f"Detected {len(result['changepoints'])} changepoints")
    print(f"Threshold: {result['meta']['threshold']:.3f}")


def demo_moving_window():
    """Demo: Moving window detector."""
    print("\n=== Moving Window Detector ===")

    detector = MovingWindowDetector(window_size=20)

    # Fit and predict
    X = np.random.randn(200, 2)
    detector.fit(X)
    result = detector.predict(X)

    print(f"Detected {len(result['changepoints'])} changepoints")
    print(f"Window size: {result['meta']['window_size']}")


if __name__ == "__main__":
    demo_simple_detector()
    demo_moving_window()
