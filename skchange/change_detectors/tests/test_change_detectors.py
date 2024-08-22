"""Basic tests for all change detectors."""

import pandas as pd
import pytest

from skchange.change_detectors.moscore import Moscore
from skchange.change_detectors.pelt import Pelt
from skchange.change_detectors.seeded_binseg import SeededBinarySegmentation
from skchange.datasets.generate import generate_teeth_data

change_detectors = [Moscore, Pelt, SeededBinarySegmentation]


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_sparse_int(Estimator):
    """Test sparse int_label segmentation."""
    n_segments = 2
    seg_len = 50
    df = generate_teeth_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator.create_test_instance()
    detector.set_params(fmt="sparse", labels="int_label")
    changepoints = detector.fit_predict(df)
    assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len - 1


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_sparse_indicator(Estimator):
    """Test sparse indicator segmentation."""
    n_segments = 2
    seg_len = 50
    df = generate_teeth_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=3
    )
    detector = Estimator.create_test_instance()
    detector.set_params(fmt="sparse", labels="indicator")
    changepoints = detector.fit_predict(df)
    assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len - 1


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_score(Estimator):
    """Test sparse score segmentation."""
    n_segments = 2
    seg_len = 50
    df = generate_teeth_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=4
    )
    sparse_detector = Estimator.create_test_instance()
    sparse_detector.set_params(fmt="sparse", labels="score")
    dense_detector = Estimator.create_test_instance()
    dense_detector.set_params(fmt="dense", labels="score")
    sparse_scores = sparse_detector.fit_predict(df)
    dense_scores = dense_detector.fit_predict(df)
    assert (sparse_scores == dense_scores).all(axis=None)
    if isinstance(sparse_scores, pd.DataFrame):
        assert "score" in sparse_scores.columns
    else:
        assert sparse_scores.name == "score"


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_dense_int(Estimator):
    """Tests dense int_label segmentation."""
    n_segments = 2
    seg_len = 50
    df = generate_teeth_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator.create_test_instance()
    detector.set_params(fmt="dense", labels="int_label")
    labels = detector.fit_predict(df)
    assert labels.nunique() == n_segments
    assert labels[seg_len - 1] == 0.0 and labels[seg_len] == 1.0


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_dense_indicator(Estimator):
    """Tests dense indicator segmentation."""
    n_segments = 2
    seg_len = 50
    df = generate_teeth_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=8
    )
    detector = Estimator.create_test_instance()
    detector.set_params(fmt="dense", labels="indicator")
    cpt_indicator = detector.fit_predict(df)
    assert cpt_indicator.sum() == n_segments - 1
    assert cpt_indicator[seg_len - 1]
