"""Tests for ContinuousLinearTrendScore with change detectors."""

import numpy as np
import pandas as pd
import pytest

from skchange.change_detectors import MovingWindow, SeededBinarySegmentation
from skchange.change_scores import ContinuousLinearTrendScore


def generate_continuous_piecewise_linear_signal(
    change_points, slopes, intercept=0, n_samples=200, noise_std=0.1, random_seed=None
):
    """Generate a continuous piecewise linear signal with noise.

    Parameters
    ----------
    change_points : list
        List of indices where the slope changes (kink points)
    slopes : list
        List of slopes for each segment (should be one more than change_points)
    intercept : float, default=0
        Starting intercept value
    n_samples : int, default=200
        Total number of samples
    noise_std : float, default=0.1
        Standard deviation of the Gaussian noise to add
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with the signal and corresponding time points
    list
        List of true change points (as indices)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if len(slopes) != len(change_points) + 1:
        raise ValueError(
            "Number of slopes must be one more than number of change points"
        )

    # Create time points and allocate signal
    time = np.arange(n_samples)
    signal = np.zeros(n_samples)

    # First segment
    signal[: change_points[0]] = intercept + slopes[0] * time[: change_points[0]]
    current_value = signal[change_points[0] - 1]

    # Middle segments
    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i + 1]
        segment_time = time[start_idx:end_idx] - time[start_idx]
        signal[start_idx:end_idx] = current_value + slopes[i + 1] * segment_time
        current_value = signal[end_idx - 1]

    # Last segment
    if len(change_points) > 0:
        last_start = change_points[-1]
        segment_time = time[last_start:] - time[last_start]
        signal[last_start:] = current_value + slopes[-1] * segment_time

    # Add noise
    signal += np.random.normal(0, noise_std, n_samples)

    # Convert to DataFrame
    df = pd.DataFrame({"signal": signal})

    return df


def test_moving_window_single_changepoint():
    """Test MovingWindow with ContinuousLinearTrendScore on a single changepoint."""
    # Generate data with a single changepoint at position 100
    true_change_points = [100]
    slopes = [0.1, -0.2]  # Positive slope followed by negative slope
    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=200,
        noise_std=1.5,
        random_seed=42,
    )

    # Create detector with ContinuousLinearTrendScore
    detector = MovingWindow(
        change_score=ContinuousLinearTrendScore(),
        bandwidth=30,
        penalty=25,  # Tuned for this specific test case
    )

    # Fit and predict
    detected_cps = detector.fit_predict(df)

    # Assert we found 1 changepoint
    assert len(detected_cps) == 1, f"Expected 1 changepoint, found {len(detected_cps)}"

    # Assert the changepoint is close to the true changepoint
    detected_cp = detected_cps.iloc[0, 0]
    cp_detection_margin = 3
    assert abs(detected_cp - true_change_points[0]) <= cp_detection_margin, (
        f"Detected {detected_cp}, expected close to {true_change_points[0]}"
    )


def test_moving_window_multiple_changepoints():
    """Test MovingWindow with ContinuousLinearTrendScore on multiple changepoints."""
    # Generate data with multiple changepoints
    true_change_points = [100, 200, 300]
    slopes = [0.1, -0.2, 0.15, -0.1]
    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=400,
        noise_std=1.5,
        random_seed=42,
    )

    # Create detector with ContinuousLinearTrendScore
    detector = MovingWindow(
        change_score=ContinuousLinearTrendScore(), bandwidth=50, penalty=20
    )

    # Fit and predict
    detected_cps = detector.fit_predict(df)

    # Assert we found the correct number of changepoints
    assert len(detected_cps) == len(true_change_points), (
        f"Expected {len(true_change_points)} changepoints, found {len(detected_cps)}"
    )

    # Assert the changepoints are close to the true changepoints
    cp_detection_margin = 3
    for i, cp in enumerate(detected_cps["ilocs"]):
        assert abs(cp - true_change_points[i]) <= cp_detection_margin, (
            f"Detected {cp}, expected close to {true_change_points[i]}"
        )


def test_seeded_binseg_single_changepoint():
    """Test SeededBinarySegmentation with ContinuousLinearTrendScore

    On a single changepoint.
    """
    # Generate data with a single changepoint
    true_change_points = [100]
    slopes = [0.1, -0.2]
    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=200,
        noise_std=1.5,
        random_seed=42,
    )

    # Create detector with ContinuousLinearTrendScore
    detector = SeededBinarySegmentation(
        change_score=ContinuousLinearTrendScore(), penalty=25, min_segment_length=20
    )

    # Fit and predict
    detected_cps = detector.fit_predict(df)

    # Assert we found 1 changepoint
    assert len(detected_cps) == 1, f"Expected 1 changepoint, found {len(detected_cps)}"

    # Assert the changepoint is close to the true changepoint
    detected_cp = detected_cps.iloc[0, 0]
    cp_detection_margin = 3
    assert abs(detected_cp - true_change_points[0]) <= cp_detection_margin, (
        f"Detected {detected_cp}, expected close to {true_change_points[0]}"
    )


def test_seeded_binseg_multiple_changepoints():
    """Test SeededBinarySegmentation with ContinuousLinearTrendScore

    On multiple changepoints.
    """
    # Generate data with multiple changepoints
    true_change_points = [100, 200, 300]
    slopes = [0.1, -0.2, 0.15, -0.1]
    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=400,
        noise_std=2.0,
        random_seed=42,
    )

    # Create detector with ContinuousLinearTrendScore
    detector = SeededBinarySegmentation(
        change_score=ContinuousLinearTrendScore(), penalty=50, min_segment_length=5
    )

    # Fit and predict
    detected_cps = detector.fit_predict(df)

    # Assert we found the correct number of changepoints
    assert len(detected_cps) == len(true_change_points), (
        f"Expected {len(true_change_points)} changepoints, found {len(detected_cps)}"
    )

    # Assert the changepoints are close to the true changepoints
    cp_detection_margin = 4
    for i, cp in enumerate(detected_cps["ilocs"]):
        assert abs(cp - true_change_points[i]) <= cp_detection_margin, (
            f"Detected {cp}, expected close to {true_change_points[i]}"
        )


def test_noise_sensitivity():
    """Test the sensitivity of both algorithms to different noise levels."""
    # Generate data with a single changepoint
    true_cps = [100]
    slopes = [0.1, -0.2]

    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
    max_deviations = []

    for noise_std in noise_levels:
        df = generate_continuous_piecewise_linear_signal(
            change_points=true_cps,
            slopes=slopes,
            n_samples=200,
            noise_std=noise_std,
            random_seed=42,
        )

        # Test MovingWindow
        mw_detector = MovingWindow(
            change_score=ContinuousLinearTrendScore(), bandwidth=30, penalty=25
        )

        mw_cps = mw_detector.fit_predict(df)

        if len(mw_cps) == 1:
            deviation = abs(mw_cps.iloc[0, 0] - true_cps[0])
            max_deviations.append(deviation)

    # Assert that for reasonable noise levels, we can detect the changepoint
    assert len(max_deviations) == len(noise_levels), (
        "Detection worked for low noise levels"
    )
    assert max(max_deviations) < 3, (
        f"Detection failed, with max deviation: {max(max_deviations)}"
    )
    # For lower noise levels, the detection should be more accurate
    assert all(np.diff(max_deviations) >= 0), (
        "Detection accuracy doesn't improve with lower noise"
    )


def test_multivariate_detection():
    """Test detection on multivariate continuous piecewise linear signals."""
    # Generate two different signals with the same changepoints
    change_points = [100, 200]
    slopes1 = [0.1, -0.2, 0.15]
    slopes2 = [0.05, 0.15, -0.1]

    df1 = generate_continuous_piecewise_linear_signal(
        change_points=change_points,
        slopes=slopes1,
        n_samples=300,
        noise_std=0.1,
        random_seed=42,
    )

    df2 = generate_continuous_piecewise_linear_signal(
        change_points=change_points,
        slopes=slopes2,
        n_samples=300,
        noise_std=0.1,
        random_seed=43,
    )

    # Combine into multivariate DataFrame
    df = pd.DataFrame({"signal1": df1["signal"], "signal2": df2["signal"]})

    # Test with SeededBinarySegmentation
    detector = SeededBinarySegmentation(
        change_score=ContinuousLinearTrendScore(), penalty=25, min_segment_length=5
    )

    detected_cps = detector.fit_predict(df)

    # Assert we found the correct number of changepoints
    assert len(detected_cps) == len(change_points), (
        f"Expected {len(change_points)} changepoints, found {len(detected_cps)}"
    )

    # Assert the changepoints are close to the true changepoints
    cp_detection_margin = 3
    for i, cp in enumerate(detected_cps["ilocs"]):
        assert abs(cp - change_points[i]) <= cp_detection_margin, (
            f"Detected {cp}, expected close to {change_points[i]}"
        )


true_change_points = [100]
slopes = [0.1, -0.2]
n_samples = 200

df = generate_continuous_piecewise_linear_signal(
    change_points=true_change_points,
    slopes=slopes,
    n_samples=n_samples,
    noise_std=0.1,
    random_seed=42,
)

# Create irregular time sampling by selectively removing points
# Keep all points around the changepoint for accurate detection
np.random.seed(42)
selection_mask = np.ones(n_samples, dtype=bool)

# Remove ~30% of points away from the changepoint
for region in [(0, 80), (120, 200)]:
    start, end = region
    region_len = end - start
    # Remove about 30% of points in each region
    to_remove = np.random.choice(
        np.arange(start, end), size=int(region_len * 0.3), replace=False
    )
    selection_mask[to_remove] = False

# Apply the mask to create irregularly sampled data
irregular_df = df[selection_mask].copy()

# Create a sample_times column that reflects the original indices
irregular_df["sample_times"] = 50.0 + 2.0 * np.arange(n_samples)[selection_mask]

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.plot(irregular_df["sample_times"], irregular_df["signal"], label="Irregular Signal")


def test_irregular_time_sampling():
    """Test ContinuousLinearTrendScore with irregular time sampling."""
    # Generate data with a single changepoint
    true_change_points = [100]
    slopes = [0.1, -0.2]
    n_samples = 200

    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=n_samples,
        noise_std=1.0,
        random_seed=42,
    )
    df["sample_times"] = 50.0 + 2.0 * np.arange(n_samples)

    # Create irregular time sampling by selectively removing points
    # Keep all points around the changepoint for accurate detection
    np.random.seed(42)
    selection_mask = np.ones(n_samples, dtype=bool)

    # Remove ~30% of points away from the changepoint
    for region in [(0, 80), (90, 110), (120, 200)]:
        start, end = region
        region_len = end - start
        # Remove about 30% of points in each region
        to_remove = np.random.choice(
            np.arange(start, end), size=int(region_len * 0.3), replace=False
        )
        selection_mask[to_remove] = False

    # Apply the mask to create irregularly sampled data:
    irregular_df = df[selection_mask].copy()
    reverse_index_map = df.index[selection_mask]
    # Create a sample_times column that reflects the original indices
    true_change_point_time = df["sample_times"].iloc[true_change_points[0]]

    # Create detector with ContinuousLinearTrendScore with time_column
    detector = MovingWindow(
        change_score=ContinuousLinearTrendScore(time_column="sample_times"),
        bandwidth=30,
        penalty=25,
    )

    # Fit and predict
    detected_cps = detector.fit_predict(irregular_df)

    # Assert we found 1 changepoint
    assert len(detected_cps) == 1, f"Expected 1 changepoint, found {len(detected_cps)}"

    # Get the original index that corresponds to the detected changepoint
    detected_cp_idx = reverse_index_map[detected_cps.iloc[0, 0]]
    detected_cp_time = df["sample_times"].iloc[detected_cp_idx]

    # Assert the detected time is close to the true changepoint
    cp_detection_margin = 2  # Slightly larger margin for irregular sampling
    cp_time_detection_margin = 4.0
    assert abs(detected_cp_idx - true_change_points[0]) <= cp_detection_margin, (
        f"Detection index {detected_cp_idx}, expected close to {true_change_points[0]}"
    )
    # Assert the detected time is close to the true changepoint
    assert abs(detected_cp_time - true_change_point_time) <= cp_time_detection_margin, (
        f"Detected time {detected_cp_time}, expected close \
        to {true_change_point_time}"
    )

    # Test with SeededBinarySegmentation as well
    sbs_detector = SeededBinarySegmentation(
        change_score=ContinuousLinearTrendScore(time_column="sample_times"),
        penalty=25,
        min_segment_length=10,
    )

    # Fit and predict
    sbs_detected_cps = sbs_detector.fit_predict(irregular_df)

    # Assert we found 1 changepoint
    assert len(sbs_detected_cps) == 1, (
        f"Expected 1 changepoint, found {len(sbs_detected_cps)}"
    )

    # Get the original time that corresponds to the detected changepoint
    sbs_detected_cp_idx = reverse_index_map[sbs_detected_cps.iloc[0, 0]]
    sbs_detected_cp_time = df["sample_times"].iloc[sbs_detected_cp_idx]

    # Assert the detected time is close to the true changepoint
    assert abs(sbs_detected_cp_idx - true_change_points[0]) <= cp_detection_margin, (
        f"SBS detected at time {sbs_detected_cp_time}, expected close \
          to {true_change_points[0]}"
    )
    # Assert the detected time is close to the true changepoint
    assert (
        abs(sbs_detected_cp_time - true_change_point_time) <= cp_time_detection_margin
    ), (
        f"SBS detected time {sbs_detected_cp_time}, expected close \
          to {true_change_point_time}"
    )


def test_ignoring_irregular_time_sampling():
    """Test ContinuousLinearTrendScore when ignoring irregular time sampling."""
    # Generate data with a single changepoint
    true_change_points = [100]
    slopes = [0.1, -0.2]
    n_samples = 200

    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=n_samples,
        noise_std=1.0,
        random_seed=42,
    )
    # df["sample_times"] = 50.0 + 2.0 * np.arange(n_samples)

    # Create irregular time sampling by selectively removing points
    # Keep all points around the changepoint for accurate detection
    np.random.seed(42)
    selection_mask = np.ones(n_samples, dtype=bool)

    # Remove ~30% of points away from the changepoint
    for region in [(0, 80), (90, 110), (120, 200)]:
        start, end = region
        region_len = end - start
        # Remove about 30% of points in each region
        to_remove = np.random.choice(
            np.arange(start, end), size=int(region_len * 0.3), replace=False
        )
        selection_mask[to_remove] = False

    # Apply the mask to create irregularly sampled data:
    irregular_df = df[selection_mask].copy()
    reverse_index_map = df.index[selection_mask]

    # Closest index to the true changepoint in the original data:
    true_cp_index = np.where(selection_mask[: true_change_points[0] + 1])[0][-1]

    # Create detector with ContinuousLinearTrendScore WITHOUT time_column
    mw_detector = MovingWindow(
        change_score=ContinuousLinearTrendScore(),  # No time_column provided
        bandwidth=30,
        penalty=25,
    )

    # Fit and predict
    mw_detected_cps = mw_detector.fit_predict(irregular_df)

    # Assert we found 1 changepoint
    assert len(mw_detected_cps) == 1, (
        f"Expected 1 changepoint, found {len(mw_detected_cps)}"
    )

    # Get the detected changepoint index in the irregular data
    mw_detected_cp_idx = reverse_index_map[mw_detected_cps.iloc[0, 0]]

    # Without time information, the detected changepoint index should be different
    # from the one we would expect with time information
    lower_cp_detection_margin = 2
    upper_cp_detection_margin = 12
    assert (
        lower_cp_detection_margin
        < abs(mw_detected_cp_idx - true_cp_index)
        < upper_cp_detection_margin
    ), f"Detection could fail when ignoring irregular sampling: {mw_detected_cp_idx}"

    # Test with SeededBinarySegmentation as well
    sbs_detector = SeededBinarySegmentation(
        change_score=ContinuousLinearTrendScore(),  # No time_column provided
        penalty=25,
        min_segment_length=10,
    )

    # Fit and predict
    sbs_detected_cps = sbs_detector.fit_predict(irregular_df)

    # Assert we found 1 changepoint
    assert len(sbs_detected_cps) == 1, (
        f"Expected 1 changepoint, found {len(sbs_detected_cps)}"
    )

    # Get the detected changepoint index in the irregular data
    sbs_detected_cp_idx = reverse_index_map[sbs_detected_cps.iloc[0, 0]]

    # Without time information, the detected changepoint index should be different
    assert (
        lower_cp_detection_margin
        < abs(sbs_detected_cp_idx - true_cp_index)
        < upper_cp_detection_margin
    ), (
        f"SBS detection should be less accurate without time information, \
        but got index {sbs_detected_cp_idx}"
    )
