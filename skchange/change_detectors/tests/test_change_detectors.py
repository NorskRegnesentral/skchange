"""Basic tests for all change detectors."""

import numpy as np
import pandas as pd
import pytest
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.annotation import make_annotation_problem

from skchange.change_detectors.binary_segmentation import SeededBinarySegmentation
from skchange.change_detectors.moscore import Moscore
from skchange.change_detectors.pelt import Pelt
from skchange.datasets.generate import teeth

change_detectors = [Moscore, Pelt, SeededBinarySegmentation]


@pytest.mark.parametrize("Estimator", change_detectors)
def test_output_type(Estimator):
    """Test annotator output type."""
    estimator = Estimator.create_test_instance()
    if not run_test_for_class(Estimator):
        return None

    arg = make_annotation_problem(
        n_timepoints=50, estimator_type=estimator.get_tag("distribution_type")
    )
    estimator.fit(arg)
    arg = make_annotation_problem(
        n_timepoints=30, estimator_type=estimator.get_tag("distribution_type")
    )
    y_pred = estimator.predict(arg)
    assert isinstance(y_pred, (pd.Series, np.ndarray))


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_sparse_int(Estimator):
    """Test sparse int_label segmentation."""
    n_segments = 2
    seg_len = 50
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator(fmt="sparse", labels="int_label")
    changepoints = detector.fit_predict(df)
    assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len - 1


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_sparse_indicator(Estimator):
    """Test sparse indicator segmentation."""
    n_segments = 2
    seg_len = 50
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=3
    )
    detector = Estimator(fmt="sparse", labels="indicator")
    changepoints = detector.fit_predict(df)
    assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len - 1


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_sparse_score(Estimator):
    """Test sparse score segmentation."""
    n_segments = 2
    seg_len = 50
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=4
    )
    detector = Estimator(fmt="sparse", labels="score")
    scores = detector.fit_predict(df)
    assert len(scores) == n_segments - 1 and scores.index[0] == seg_len - 1
    assert np.all(scores >= 0.0)


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_dense_int(Estimator):
    """Tests dense int_label segmentation."""
    n_segments = 2
    seg_len = 50
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator(fmt="dense", labels="int_label")
    labels = detector.fit_predict(df)
    assert labels.nunique() == n_segments
    assert labels[seg_len - 1] == 0.0 and labels[seg_len] == 1.0


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_dense_indicator(Estimator):
    """Tests dense indicator segmentation."""
    n_segments = 2
    seg_len = 50
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=8
    )
    detector = Estimator(fmt="dense", labels="indicator")
    cpt_indicator = detector.fit_predict(df)
    assert cpt_indicator.sum() == n_segments - 1
    assert cpt_indicator[seg_len - 1]


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_dense_score(Estimator):
    """Tests dense score segmentation."""
    n_segments = 2
    seg_len = 50
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator(fmt="dense", labels="score")
    scores = detector.fit_predict(df)
    assert scores.size == df.shape[0]
    assert np.all(scores >= 0.0)
