"""Examples showing flexible y parameter handling.

The design supports multiple interpretations of y based on structure.

Following the sparse-first principle, Segmentation dicts are the preferred
format for y labels when changepoint locations are known.
"""

import numpy as np

from skchange.new_api.base import BaseChangeDetector
from skchange.new_api.typing import ArrayLike, Segmentation
from skchange.new_api.utils import dense_to_sparse, make_segmentation


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

    def _predict(self, X: ArrayLike) -> Segmentation:
        """Predict on single series."""
        # Dummy prediction
        n = len(X)
        cps = np.array([n // 2])
        return make_segmentation(
            changepoints=cps,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            meta={"mode": self.mode_},
        )


# ============================================================================
# Usage Examples
# ============================================================================


def example_sparse_segmentation_labels():
    """Sparse-first: Using Segmentation dicts for y (preferred)."""
    print("=" * 70)
    print("Example 0: Sparse Segmentation Labels (Preferred)")
    print("=" * 70)

    detector = FlexibleDetector()

    # Single series with known changepoints (sparse format)
    print("\nSingle series with sparse labels:")
    X = np.random.randn(150, 3)
    y_sparse = make_segmentation(
        changepoints=np.array([50, 100]),
        labels=np.array([0, 1, 2]),
        n_samples=150,
    )
    detector.fit(X, y_sparse)
    print(f"  Fitted with sparse labels: {y_sparse['changepoints']}")
    print(f"  Mode: {detector.mode_}")
    print("  Automatically converted to dense internally")

    # Multiple series with sparse labels
    print("\nMultiple series with sparse labels:")
    X_list = [
        np.random.randn(100, 3),
        np.random.randn(80, 3),
        np.random.randn(120, 3),
    ]
    y_sparse_list = [
        make_segmentation(
            changepoints=np.array([40]),
            n_samples=100,
        ),
        make_segmentation(
            changepoints=np.array([30, 60]),
            n_samples=80,
        ),
        make_segmentation(
            changepoints=np.array([50, 90]),
            n_samples=120,
        ),
    ]
    detector.fit(X_list, y_sparse_list)
    print(f"  Fitted {len(X_list)} series with sparse labels")
    print(f"  Mode: {detector.mode_}")
    print("  Series 1:", y_sparse_list[0]["changepoints"])
    print("  Series 2:", y_sparse_list[1]["changepoints"])
    print("  Series 3:", y_sparse_list[2]["changepoints"])


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
    """Per-timepoint labels - must convert dense to sparse first."""
    print("\n" + "=" * 70)
    print("Example 2: Per-Timepoint Labels (Dense -> Sparse)")
    print("=" * 70)

    detector = FlexibleDetector()

    # Single series with dense labels - convert to sparse
    print("\nSingle series (dense labels converted to sparse):")
    X = np.random.randn(100, 3)
    y_dense = np.repeat([0, 1, 0], [30, 40, 30])  # Dense segment labels
    y_sparse = dense_to_sparse(y_dense)  # Convert to Segmentation
    detector.fit(X, y_sparse)
    print(f"  Dense labels shape: {y_dense.shape}")
    print(f"  Converted to sparse: {y_sparse['changepoints']}")
    print(f"  Mode: {detector.mode_}")

    # Multiple series with sparse labels
    print("\nMultiple series with sparse labels:")
    X_list = [
        np.random.randn(100, 3),
        np.random.randn(50, 3),
        np.random.randn(75, 3),
    ]
    # Convert dense labels to sparse for each series
    y_dense_list = [
        np.repeat([0, 1, 0], [30, 40, 30]),
        np.repeat([0, 1], [25, 25]),
        np.repeat([0, 1, 2], [25, 25, 25]),
    ]
    y_list = [dense_to_sparse(y_d) for y_d in y_dense_list]
    detector.fit(X_list, y_list)
    print(f"  Fitted {len(X_list)} series with sparse labels")
    print(f"  Mode: {detector.mode_}")


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
    # Known changepoints as sparse Segmentation
    y_train = make_segmentation(
        changepoints=np.array([100, 500, 800]),
        n_samples=1000,
    )
    detector.fit(X_train, y_train)
    print(f"  Trained with {len(y_train['changepoints'])} known changepoints")

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
    example_sparse_segmentation_labels()
    example_unsupervised()
    example_per_timepoint_labels()
    example_series_level_labels()
    example_real_world_use_cases()
    example_validation_errors()

    print("\n" + "=" * 70)
    print("Summary: y Parameter Flexibility")
    print("=" * 70)
    print("""
The design supports (SPARSE-FIRST):

0. Sparse Segmentation dict (y is dict) - REQUIRED for segment labels
   Example: make_segmentation(changepoints=[50, 100], n_samples=150)
1. No labels (y=None) - most common, unsupervised
2. Sparse per-series labels (y is list[Segmentation]) - segment labels
3. Series-level labels (y is 1D ArrayLike, len=n_series) - classification

Dense labels NO LONGER ACCEPTED - use dense_to_sparse() to convert:
  y_dense = np.array([0,0,0,1,1,1,2,2])
  y_sparse = dense_to_sparse(y_dense)
  detector.fit(X, y_sparse)

All formats handled automatically based on y structure!
    """)
