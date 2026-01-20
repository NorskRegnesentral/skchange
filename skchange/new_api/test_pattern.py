"""Tests demonstrating the single/multi series pattern."""

import numpy as np
import pytest

from .examples import BatchPELT, MovingWindowDetector, SimplePELT
from .typing import ChangeDetectionResult


class TestSingleSeriesOnly:
    """Test single-series only detector (SimplePELT)."""

    def test_accepts_single_series(self):
        """Single-series detector should work with single series."""
        detector = SimplePELT(penalty=1.0)

        # 2D input
        X = np.random.randn(100, 3)
        detector.fit(X)
        results = detector.predict(X)

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], ChangeDetectionResult)

    def test_accepts_1d_input(self):
        """Should auto-convert 1D to 2D."""
        detector = SimplePELT(penalty=1.0)

        # 1D input - should be converted to (100, 1)
        X = np.random.randn(100)
        detector.fit(X)
        results = detector.predict(X)

        assert isinstance(results, list)
        assert len(results) == 1

    def test_rejects_multiple_series(self):
        """Single-series detector should reject list input."""
        detector = SimplePELT(penalty=1.0)

        X_list = [np.random.randn(100, 2), np.random.randn(50, 2)]

        with pytest.raises(ValueError, match="does not support multiple series"):
            detector.fit(X_list)

    def test_rejects_3d_input(self):
        """Should reject 3D arrays."""
        detector = SimplePELT(penalty=1.0)

        X = np.random.randn(100, 3, 5)  # 3D

        with pytest.raises(ValueError, match="Expected 1D or 2D"):
            detector.fit(X)


class TestUniversalDetector:
    """Test universal detector (MovingWindowDetector)."""

    def test_single_series(self):
        """Should work on single series."""
        detector = MovingWindowDetector(window_size=20)

        X = np.random.randn(100, 2)
        detector.fit(X)
        results = detector.predict(X)

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], ChangeDetectionResult)

    def test_multiple_series(self):
        """Should work on multiple series."""
        detector = MovingWindowDetector(window_size=20)

        X_list = [
            np.random.randn(100, 2),
            np.random.randn(150, 2),
            np.random.randn(200, 3),  # Different number of channels is OK
        ]

        detector.fit(X_list)
        results = detector.predict(X_list)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, ChangeDetectionResult) for r in results)

    def test_transform_single(self):
        """Transform should work on single series."""
        detector = MovingWindowDetector(window_size=20)

        X = np.random.randn(100, 2)
        detector.fit(X)
        labels_list = detector.transform(X)

        assert isinstance(labels_list, list)
        assert len(labels_list) == 1
        assert len(labels_list[0]) == len(X)

    def test_transform_multiple(self):
        """Transform should work on multiple series."""
        detector = MovingWindowDetector(window_size=20)

        X_list = [np.random.randn(100, 2), np.random.randn(50, 2)]

        detector.fit(X_list)
        labels_list = detector.transform(X_list)

        assert isinstance(labels_list, list)
        assert len(labels_list) == 2
        assert len(labels_list[0]) == 100
        assert len(labels_list[1]) == 50


class TestBatchOptimizedDetector:
    """Test batch-optimized detector (BatchPELT)."""

    def test_learns_from_multiple_series(self):
        """Should learn shared parameters from multiple series."""
        detector = BatchPELT(penalty=None)  # Auto-tune

        X_list = [
            np.random.randn(100, 1),
            np.random.randn(150, 1),
            np.random.randn(200, 1),
        ]

        detector.fit(X_list)

        # Should have learned global threshold
        assert detector.global_threshold_ is not None
        assert detector.penalty is not None

        # Should use same threshold for all predictions
        results = detector.predict(X_list)
        assert len(results) == 3

    def test_works_on_single_series(self):
        """Should also work on single series (fallback)."""
        detector = BatchPELT(penalty=2.0)

        X = np.random.randn(100, 1)
        detector.fit(X)
        results = detector.predict(X)

        assert isinstance(results, list)
        assert len(results) == 1
        assert detector.global_threshold_ is not None


class TestInputValidation:
    """Test input validation and normalization."""

    def test_1d_to_2d_conversion(self):
        """1D arrays should be converted to (n, 1)."""
        detector = MovingWindowDetector(window_size=10)

        X_1d = np.random.randn(100)
        detector.fit(X_1d)

        # Should work without errors
        results = detector.predict(X_1d)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_inconsistent_y_type_error(self):
        """If X is list, y must also be list or None."""
        detector = MovingWindowDetector(window_size=10)

        X_list = [np.random.randn(100, 2), np.random.randn(50, 2)]
        y_single = np.zeros(100)  # Wrong - not a list

        with pytest.raises(ValueError, match="y must also be a list"):
            detector.fit(X_list, y_single)

    def test_different_length_series(self):
        """Multiple series can have different lengths."""
        detector = MovingWindowDetector(window_size=10)

        X_list = [
            np.random.randn(100, 2),
            np.random.randn(50, 2),
            np.random.randn(200, 2),
        ]

        detector.fit(X_list)
        results = detector.predict(X_list)

        assert len(results) == 3


class TestTagSystem:
    """Test tag-based capability system."""

    def test_get_tag(self):
        """Should retrieve tags correctly."""
        detector = SimplePELT()
        assert detector.get_tag("capability:multiple_series") is False

        detector = MovingWindowDetector()
        assert detector.get_tag("capability:multiple_series") is True

    def test_tag_controls_behavior(self):
        """Tag should control whether multiple series are accepted."""
        # Single-series detector
        single = SimplePELT()
        assert single.get_tag("capability:multiple_series") is False

        with pytest.raises(ValueError):
            single.fit([np.random.randn(100, 2)])

        # Multi-series detector
        multi = MovingWindowDetector()
        assert multi.get_tag("capability:multiple_series") is True

        # Should work
        multi.fit([np.random.randn(100, 2)])


def test_end_to_end_workflow():
    """Test complete workflow: fit → predict → transform."""
    # Generate synthetic data with changepoint
    np.random.seed(42)
    X = np.concatenate([np.random.randn(50, 2), np.random.randn(50, 2) + 3])

    # Single-series workflow
    detector = MovingWindowDetector(window_size=10)
    detector.fit(X)

    # Predict (sparse)
    results = detector.predict(X)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], ChangeDetectionResult)

    # Transform (dense)
    labels_list = detector.transform(X)
    assert len(labels_list) == 1
    assert len(labels_list[0]) == len(X)
    assert labels_list[0].dtype == int

    # Multiple-series workflow
    X_list = [X, X[:60], X[40:]]

    detector.fit(X_list)
    results = detector.predict(X_list)

    assert len(results) == 3

    labels_list = detector.transform(X_list)
    assert len(labels_list) == 3
    assert len(labels_list[0]) == 100
    assert len(labels_list[1]) == 60
    assert len(labels_list[2]) == 60


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
