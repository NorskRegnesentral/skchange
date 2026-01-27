"""Example usage of metrics and model selection for changepoint detection."""

import numpy as np

from skchange.new_api.examples import MovingWindowDetector
from skchange.new_api.metrics import adjusted_rand_index, f1_score, hausdorff_metric
from skchange.new_api.model_selection import (
    SeriesKFold,
    cross_val_score,
    train_test_split,
)


def example_single_series_evaluation():
    """Demonstrate detector evaluation on single series."""
    print("=" * 60)
    print("Single Series Evaluation")
    print("=" * 60)

    # Generate synthetic data with changepoints
    np.random.seed(42)
    X = np.concatenate(
        [
            np.random.randn(50, 2),
            np.random.randn(50, 2) + 2,
            np.random.randn(50, 2) - 1,
        ]
    )

    # Ground truth changepoints
    y_true_indices = np.array([50, 100])
    y_true_labels = np.repeat([0, 1, 2], 50)

    # Fit and predict
    detector = MovingWindowDetector(window_size=10)
    detector.fit(X)
    result = detector.predict(X)

    # Evaluate with changepoint metric
    score_hausdorff = hausdorff_metric(y_true_indices, result)
    score_f1 = f1_score(y_true_indices, result, tolerance=5)

    print(f"Hausdorff distance: {score_hausdorff:.2f}")
    print(f"F1 score (tol=5): {score_f1:.3f}")

    # Evaluate with segment metric
    score_ari = adjusted_rand_index(y_true_labels, result)
    print(f"Adjusted Rand Index: {score_ari:.3f}")

    print()


def example_multiple_series_evaluation():
    """Demonstrate detector evaluation on multiple series."""
    print("=" * 60)
    print("Multiple Series Evaluation")
    print("=" * 60)

    # Generate multiple series
    np.random.seed(42)
    X_list = [
        np.concatenate([np.random.randn(40, 2), np.random.randn(60, 2) + 2]),
        np.concatenate(
            [
                np.random.randn(30, 2),
                np.random.randn(30, 2) + 1.5,
                np.random.randn(40, 2) - 1,
            ]
        ),
        np.concatenate([np.random.randn(50, 2), np.random.randn(50, 2) + 3]),
    ]

    # Ground truth changepoints for each series
    y_true_list = [
        np.array([40]),
        np.array([30, 60]),
        np.array([50]),
    ]

    # Fit on all series
    detector = MovingWindowDetector(window_size=10)
    detector.fit(X_list)

    # Predict on each series
    results = [detector.predict(X) for X in X_list]

    # Evaluate - use list comprehension for aggregation
    scores_hausdorff = [
        hausdorff_metric(yt, result) for yt, result in zip(y_true_list, results)
    ]
    scores_f1 = [
        f1_score(yt, result, tolerance=5) for yt, result in zip(y_true_list, results)
    ]

    print(f"Mean Hausdorff distance: {np.mean(scores_hausdorff):.2f}")
    print(f"Mean F1 score (tol=5): {np.mean(scores_f1):.3f}")

    # Per-series scores
    print("\nPer-series scores:")
    for i, (score, result) in enumerate(zip(scores_f1, results)):
        print(
            f"  Series {i + 1}: F1 = {score:.3f}, "
            f"detected {len(result['changepoints'])} changepoints"
        )

    print()


def example_cross_validation():
    """Demonstrate cross-validation for model evaluation."""
    print("=" * 60)
    print("Cross-Validation")
    print("=" * 60)

    # Generate multiple series
    np.random.seed(42)
    n_series = 10
    X_list = []
    y_true_list = []

    for i in range(n_series):
        # Random number of segments
        n_segments = np.random.randint(2, 5)
        segment_lengths = np.random.randint(30, 70, size=n_segments)

        # Generate series with shifts
        segments = [
            np.random.randn(length, 2) + np.random.randn(2) * 2
            for length in segment_lengths
        ]
        X = np.concatenate(segments)
        X_list.append(X)

        # True changepoints
        changepoints = np.cumsum(segment_lengths)[:-1]
        y_true_list.append(changepoints)

    # Cross-validation with changepoint metric
    detector = MovingWindowDetector(window_size=15)
    cv = SeriesKFold(n_splits=5, shuffle=True, random_state=42)

    scores_hausdorff = cross_val_score(
        detector, X_list, y_true_list, cv=cv, scoring=hausdorff_metric
    )

    scores_f1 = cross_val_score(
        detector,
        X_list,
        y_true_list,
        cv=cv,
        scoring=lambda yt, yp: f1_score(yt, yp, tolerance=5),
    )

    print(f"Hausdorff CV scores: {scores_hausdorff}")
    print(f"  Mean: {scores_hausdorff.mean():.2f} (+/- {scores_hausdorff.std():.2f})")
    print()
    print(f"F1 CV scores: {scores_f1}")
    print(f"  Mean: {scores_f1.mean():.3f} (+/- {scores_f1.std():.3f})")

    print()


def example_train_test_split():
    """Demonstrate train/test split for series data."""
    print("=" * 60)
    print("Train/Test Split")
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    X_list = [np.random.randn(100, 2) for _ in range(20)]
    y_true_list = [np.array([np.random.randint(30, 70)]) for _ in range(20)]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_list, y_true_list, test_size=0.3, random_state=42
    )

    print(f"Total series: {len(X_list)}")
    print(f"Training series: {len(X_train)}")
    print(f"Test series: {len(X_test)}")

    # Train and evaluate
    detector = MovingWindowDetector(window_size=10)
    detector.fit(X_train, y_train)

    # Predict on test set
    y_pred = [detector.predict(X) for X in X_test]

    # Evaluate - compute scores per series then aggregate
    scores = [f1_score(yt, yp, tolerance=5) for yt, yp in zip(y_test, y_pred)]
    mean_score = np.mean(scores)
    print(f"\nTest F1 score: {mean_score:.3f}")

    print()


if __name__ == "__main__":
    example_single_series_evaluation()
    example_multiple_series_evaluation()
    example_cross_validation()
    example_train_test_split()
