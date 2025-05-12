import numpy as np
import pytest

from skchange.change_detectors._crops import CROPS_PELT
from skchange.costs import L2Cost
from skchange.datasets import generate_alternating_data


def test_pelt_crops():
    """Test the CROPS algorithm for path solutions to penalized CPD.

    Reference: https://arxiv.org/pdf/1412.3617
    """
    cost = L2Cost()
    min_penalty = 0.5
    max_penalty = 50.0

    change_point_detector = CROPS_PELT(
        cost=cost,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
        min_segment_length=30,
    )

    # Generate test data:
    dataset = generate_alternating_data(
        n_segments=2,
        segment_length=100,
        p=1,
        mean=3.0,
        variance=4.0,
        random_state=42,
    )

    # Issues: non-restrictive pruning:
    # - min_segment_length=30
    # - high penalty: 50.0
    # - middle penalty: np.float64(1.6562305936619783)
    # - low penalty: np.float64(1.6120892290743671)

    # Issues: with restrictive pruning.
    # TODO: Look into when a difference from optimal partioning
    # occurs. First time a potential change point start is pruned,
    # that should not have been pruned.
    # - min_segment_length=30
    # high_penalty: 50.0
    # middle_penalty: np.float64(1.6562305936619783)
    # low_penalty: np.float64(1.6120892290743671)

    # Fit the change point detector:
    change_point_detector.fit(dataset)
    results = change_point_detector.run_crops(dataset.values)
    # Check that the results are as expected:
    assert len(results) == 64


def test_pelt_crops_raises_on_wrong_segmentation_selection():
    """Test CROPS algorithm raises error when segmentation selection is wrong."""
    cost = L2Cost()
    min_penalty = 1.0
    max_penalty = 2.0

    # Check that the results are as expected:
    with pytest.raises(ValueError):
        CROPS_PELT(
            cost=cost,
            segmentation_selection="wrong",
            min_penalty=min_penalty,
            max_penalty=max_penalty,
        )


def test_retrieve_change_points_before_predict_raises():
    """Test retrieve_change_points method raises an error if called before predict."""
    cost = L2Cost()
    min_penalty = 1.0
    max_penalty = 2.0

    change_point_detector = CROPS_PELT(
        cost=cost,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
    )

    # Generate test data:
    dataset = generate_alternating_data(
        n_segments=2,
        segment_length=100,
        p=1,
        mean=3.0,
        variance=4.0,
        random_state=42,
    )

    # Fit the change point detector:
    change_point_detector.fit(dataset)

    # Check that the results are as expected:
    with pytest.raises(ValueError):
        change_point_detector.retrieve_change_points(4)


def test_retrieve_change_points_non_existing_num_change_point():
    """Test that the retrieve_change_points method raises

    If that number of change points does not exist.
    """
    cost = L2Cost()
    min_penalty = 1.0
    max_penalty = 2.0

    change_point_detector = CROPS_PELT(
        cost=cost,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
    )

    # Generate test data:
    dataset = generate_alternating_data(
        n_segments=2,
        segment_length=100,
        p=1,
        mean=3.0,
        variance=4.0,
        random_state=42,
    )

    # Fit the change point detector:
    change_point_detector.fit(dataset)
    change_point_detector.run_crops(dataset)

    # Check that the results are as expected:
    with pytest.raises(ValueError):
        change_point_detector.retrieve_change_points(4)


def test_retrieve_change_points():
    """Test the retrieve_change_points method."""
    cost = L2Cost()
    min_penalty = 40.0
    max_penalty = 50.0

    change_point_detector = CROPS_PELT(
        cost=cost,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
    )

    # Generate test data:
    dataset = generate_alternating_data(
        n_segments=2,
        segment_length=100,
        p=1,
        mean=3.0,
        variance=4.0,
        random_state=42,
    )

    # Fit the change point detector:
    change_point_detector.fit(dataset)
    change_point_detector.run_crops(dataset)

    # Check that the results are as expected:
    assert (
        len(change_point_detector.retrieve_change_points(1, refine_change_points=False))
        == 1
    )


def test_retrieve_refined_change_points():
    """Test the retrieve_change_points method."""
    cost = L2Cost()
    min_penalty = 10.0
    max_penalty = 50.0

    change_point_detector = CROPS_PELT(
        cost=cost,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
        min_segment_length=10,
    )

    # Generate test data:
    dataset = generate_alternating_data(
        n_segments=3,
        segment_length=88,
        p=1,
        mean=3.0,
        variance=4.0,
        random_state=42,
    )

    # Fit the change point detector:
    change_point_detector.fit(dataset)
    change_point_detector.run_crops(dataset)

    refined_change_points = change_point_detector.retrieve_change_points(
        2, refine_change_points=True
    )

    # Check that the results are as expected:
    assert np.all(refined_change_points == np.array([88, 176])), (
        f"Expected [88, 176], got {refined_change_points}"
    )
