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
    min_penalty = 0.05
    max_penalty = 50.0
    min_segment_length = 10

    # Generate test data:
    dataset = generate_alternating_data(
        n_segments=2,
        segment_length=100,
        p=1,
        mean=3.0,
        variance=4.0,
        # random_state=42,
        random_state=241,
    )

    # Fit CROPS change point detector:
    change_point_detector = CROPS_PELT(
        cost=cost,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
        min_segment_length=min_segment_length,
    )
    change_point_detector.fit(dataset)
    results = change_point_detector.run_crops(dataset.values)

    no_pruning_change_detector = CROPS_PELT(
        cost=cost,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
        min_segment_length=min_segment_length,
        drop_pruning=True,
    )
    no_pruning_change_detector.fit(dataset)
    no_pruning_results = no_pruning_change_detector.run_crops(dataset.values)

    assert np.all(
        results == no_pruning_results
    ), f"Expected {no_pruning_results}, got {results}"
    # Check that the results are as expected:
    assert len(results) == 10


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
    assert len(change_point_detector.retrieve_change_points(1)) == 1


def test_retrieve_change_points_2():
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

    refined_change_points = change_point_detector.retrieve_change_points(2)

    # Check that the results are as expected:
    assert np.array_equal(
        refined_change_points, np.array([88, 176])
    ), f"Expected [88, 176], got {refined_change_points}"
