"""Simple PELT test."""

import numpy as np
import pandas as pd
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.annotation import make_annotation_problem

from skchange.change_detectors.pelt import Pelt
from skchange.datasets.generate import teeth


def test_output_type():
    """Test annotator output type."""
    Estimator = Pelt
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
    """Test PELT sparse segmentation.

    Check if the predicted change points match.
    """
    n_cpts = 1
    seg_len = 10
    df = teeth(
        n_segments=n_cpts + 1, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Pelt(fmt="sparse")
    cpts = detector.fit_predict(df)
    # End point also included as a changepoint
    assert len(cpts) == n_cpts and cpts.index[0] == seg_len - 1


def test_pelt_dense():
    """Tests PELT dense segmentation.

    Check if the predicted segmentation matches.
    """
    n_cpts = 1
    seg_len = 10
    df = teeth(
        n_segments=n_cpts + 1, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Pelt(fmt="dense")
    labels = detector.fit_predict(df)
    assert labels.nunique() == n_cpts + 1 and labels[9] == 0.0 and labels[10] == 1.0
