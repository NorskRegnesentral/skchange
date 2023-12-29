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
def test_change_detector_sparse(Estimator):
    """Test sparse segmentation.

    Check if the predicted change points match.
    """
    n_segments = 2
    seg_len = 50
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator(fmt="sparse")
    changepoints = detector.fit_predict(df)
    # End point also included as a changepoint
    assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len - 1


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_dense(Estimator):
    """Tests dense segmentation.

    Check if the predicted segmentation matches.
    """
    n_segments = 2
    seg_len = 50
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator(fmt="dense")
    labels = detector.fit_predict(df)
    assert labels.nunique() == n_segments
    assert labels[seg_len - 1] == 0.0 and labels[seg_len] == 1.0
