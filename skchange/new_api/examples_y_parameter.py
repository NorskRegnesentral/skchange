"""Examples showing flexible y parameter handling.

The design supports multiple interpretations of y based on structure.
"""

import numpy as np

from .base import BaseChangeDetector
from .typing import ArrayLike, ChangeDetectionResult
from .utils import make_change_detection_result


class FlexibleDetector(BaseChangeDetector):
    """Example detector showing all y parameter patterns."""

    _tags = {"capability:multiple_series": True}

    def __init__(self):
        self.mode_ = None

    def _fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit single series."""
        if y is not None:
            self.mode_ = "supervised_timepoint"
            # y has shape (n_samples,) - label for each timepoint
            assert len(y) == len(X), "y must match X length"
        else:
            self.mode_ = "unsupervised"
        return self

    def _fit_multiple(
        self, X: list[ArrayLike], y: ArrayLike | list[ArrayLike] | None = None
    ):
        """Fit multiple series with flexible y."""
        if y is None:
            # Unsupervised - most common
            self.mode_ = "unsupervised_batch"

        elif isinstance(y, list):
            # Per-timepoint labels for each series
            self.mode_ = "supervised_timepoint_batch"
            for X_i, y_i in zip(X, y):
                assert len(y_i) == len(X_i), "y[i] must match X[i] length"

        else:
            # One label per series (series-level classification)
            self.mode_ = "supervised_series_level"
            y_array = np.asarray(y)
            assert len(y_array) == len(X), "y must have one label per series"

        return self

    def _predict(self, X: ArrayLike) -> ChangeDetectionResult:
        """Predict on single series."""
        # Dummy prediction
        n = len(X)
        indices = np.array([n // 2])
        return make_change_detection_result(
            indices=indices,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            meta={"mode": self.mode_},
        )


# ============================================================================
# Usage Examples
# ============================================================================


def example_unsupervised():
    """Most common: no labels."""
    print("=" * 70)
    print("Example 1: Unsupervised (no y)")
    print("=" * 70)

    detector = FlexibleDetector()

    # Single series
    X = np.random.randn(100, 3)
    detector.fit(X)
    print(f"Single series, no labels: mode = {detector.mode_}")

    # Multiple series
    X_list = [np.random.randn(100, 3), np.random.randn(50, 3)]
    detector.fit(X_list)
    print(f"Multiple series, no labels: mode = {detector.mode_}")


def example_per_timepoint_labels():
    """Per-timepoint labels (e.g., known anomalies at specific times)."""
    print("\n" + "=" * 70)
    print("Example 2: Per-Timepoint Labels")
    print("=" * 70)

    detector = FlexibleDetector()

    # Single series with per-timepoint labels
    X = np.random.randn(100, 3)
    y = np.random.randint(0, 2, size=100)  # Binary label per timepoint
    detector.fit(X, y)
    print(f"Single series + timepoint labels: mode = {detector.mode_}")

    # Multiple series with per-timepoint labels
    X_list = [
        np.random.randn(100, 3),
        np.random.randn(50, 3),
        np.random.randn(75, 3),
    ]
    y_list = [
        np.random.randint(0, 2, size=100),  # Labels for first series
        np.random.randint(0, 2, size=50),  # Labels for second series
        np.random.randint(0, 2, size=75),  # Labels for third series
    ]
    detector.fit(X_list, y_list)
    print(f"Multiple series + timepoint labels: mode = {detector.mode_}")
    print("  Each y[i] has shape matching X[i]")


def example_series_level_labels():
    """One label per series (e.g., classify each series as healthy/faulty)."""
    print("\n" + "=" * 70)
    print("Example 3: Series-Level Labels")
    print("=" * 70)

    detector = FlexibleDetector()

    # Multiple series with one label per series
    X_list = [
        np.random.randn(100, 3),  # Healthy patient
        np.random.randn(50, 3),  # Faulty sensor
        np.random.randn(75, 3),  # Healthy patient
    ]
    y_series = np.array([0, 1, 0])  # 0=healthy, 1=faulty

    detector.fit(X_list, y_series)
    print(f"Multiple series + series labels: mode = {detector.mode_}")
    print(f"  y has shape {y_series.shape} (one label per series)")


def example_real_world_use_cases():
    """Real-world scenarios."""
    print("\n" + "=" * 70)
    print("Example 4: Real-World Use Cases")
    print("=" * 70)

    # Use case 1: Anomaly detection (unsupervised)
    print("\nUse Case 1: Unsupervised anomaly detection")
    detector = FlexibleDetector()
    sensor_data = [
        np.random.randn(1000, 5),  # Sensor 1
        np.random.randn(800, 5),  # Sensor 2 (different length)
        np.random.randn(1200, 5),  # Sensor 3
    ]
    detector.fit(sensor_data)  # No labels needed
    results = detector.predict(sensor_data)
    print(f"  Detected anomalies in {len(results)} sensors")

    # Use case 2: Semi-supervised with known changepoints
    print("\nUse Case 2: Semi-supervised with known changepoints")
    detector = FlexibleDetector()
    X_train = np.random.randn(1000, 3)
    # y indicates known changepoints (1) vs normal (0)
    y_train = np.zeros(1000)
    y_train[[100, 500, 800]] = 1  # Known changepoints
    detector.fit(X_train, y_train)
    print(f"  Trained with {y_train.sum():.0f} known changepoints")

    # Use case 3: Classify time series
    print("\nUse Case 3: Series classification")
    detector = FlexibleDetector()
    patient_data = [
        np.random.randn(500, 10),  # Patient 1
        np.random.randn(600, 10),  # Patient 2
        np.random.randn(450, 10),  # Patient 3
    ]
    outcomes = np.array([1, 0, 1])  # 1=recovered, 0=not recovered
    detector.fit(patient_data, outcomes)
    print(f"  Classified {len(patient_data)} patient time series")


def example_validation_errors():
    """Show what happens with invalid y."""
    print("\n" + "=" * 70)
    print("Example 5: Validation Errors")
    print("=" * 70)

    detector = FlexibleDetector()
    X_list = [np.random.randn(100, 3), np.random.randn(50, 3)]

    # Error 1: y has wrong length (series-level)
    print("\nError 1: Wrong number of series labels")
    try:
        y_wrong = np.array([0, 1, 2])  # 3 labels for 2 series
        detector.fit(X_list, y_wrong)
    except ValueError as e:
        print(f"  ✓ Caught: {e}")

    # Error 2: y list has wrong length
    print("\nError 2: Wrong number of timepoint label arrays")
    try:
        y_wrong = [np.zeros(100)]  # Only 1 array for 2 series
        detector.fit(X_list, y_wrong)
    except ValueError as e:
        print(f"  ✓ Caught: {e}")

    # Error 3: y is multidimensional when it should be 1D
    print("\nError 3: Series labels should be 1D")
    try:
        y_wrong = np.zeros((2, 5))  # 2D array
        detector.fit(X_list, y_wrong)
    except ValueError as e:
        print(f"  ✓ Caught: {e}")


if __name__ == "__main__":
    example_unsupervised()
    example_per_timepoint_labels()
    example_series_level_labels()
    example_real_world_use_cases()
    example_validation_errors()

    print("\n" + "=" * 70)
    print("Summary: y Parameter Flexibility")
    print("=" * 70)
    print("""
The design supports:

1. No labels (y=None) - most common, unsupervised
2. Per-timepoint labels (y matches X length) - semi-supervised
3. One label per series (y is 1D, len=n_series) - classification
4. Per-timepoint labels per series (y is list) - batch semi-supervised

All handled automatically based on y structure!
    """)
