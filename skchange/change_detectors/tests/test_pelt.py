"""Simple PELT test."""

import numpy as np
import pandas as pd
from sktime.annotation.clasp import ClaSPSegmentation
from sktime.datasets import load_gun_point_segmentation
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.annotation import make_annotation_problem

# from skchange.change_detectors.pelt import Pelt


def test_output_type():
    """Test annotator output type."""
    Estimator = ClaSPSegmentation
    estimator = Estimator.create_test_instance()
    if not run_test_for_class(Estimator):
        return None

    arg = make_annotation_problem(
        n_timepoints=50, estimator_type=estimator.get_tag("distribution_type")
    )
    estimator.fit(arg)
    arg = make_annotation_problem(
        n_timepoints=10, estimator_type=estimator.get_tag("distribution_type")
    )
    y_pred = estimator.predict(arg)
    assert isinstance(y_pred, (pd.Series, np.ndarray))


def test_pelt_sparse():
    """Test ClaSP sparse segmentation.

    Check if the predicted change points match.
    """
    # load the test dataset
    ts, period_size, cps = load_gun_point_segmentation()

    # compute a ClaSP segmentation
    clasp = ClaSPSegmentation(period_size, n_cps=1)
    clasp.fit(ts)
    found_cps = clasp.predict(ts)
    scores = clasp.predict_scores(ts)

    assert len(found_cps) == 1 and found_cps[0] == 893
    assert len(scores) == 1 and scores[0] > 0.74


def test_pelt_dense():
    """Tests ClaSP dense segmentation.

    Check if the predicted segmentation matches.
    """
    # load the test dataset
    ts, period_size, cps = load_gun_point_segmentation()

    # compute a ClaSP segmentation
    clasp = ClaSPSegmentation(period_size, n_cps=1, fmt="dense")
    clasp.fit(ts)
    segmentation = clasp.predict(ts)
    scores = clasp.predict_scores(ts)

    assert len(segmentation) == 2 and segmentation[0].right == 893
    assert np.argmax(scores) == 893
