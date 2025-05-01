from skchange.change_detectors import PELT, SeededBinarySegmentation
from skchange.change_detectors._crops import CROPS_PELT, GenericCROPS
from skchange.change_scores._from_cost import to_change_score
from skchange.costs import GaussianCost, L1Cost, L2Cost
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
    results = change_point_detector.run_crops(dataset.values)
    # Check that the results are as expected:
    assert len(results) == 64
