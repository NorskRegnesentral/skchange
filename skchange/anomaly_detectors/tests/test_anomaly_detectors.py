"""Basic tests for all anomaly detectors."""

import numpy as np
import pandas as pd
import pytest
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.annotation import make_annotation_problem

from skchange.anomaly_detectors.capa import Capa
from skchange.datasets.generate import teeth

anomaly_detectors = [Capa]


@pytest.mark.parametrize("Estimator", anomaly_detectors)
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


@pytest.mark.parametrize("Estimator", anomaly_detectors)
def test_anomaly_detector_sparse(Estimator):
    """Test sparse anomaly detector output.

    Check if the predicted anomalies match.
    """
    n_segments = 2
    seg_len = 20
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator(fmt="sparse")
    anomalies = detector.fit_predict(df)
    # End point also included as a changepoint
    assert (
        len(anomalies) == 1
        and anomalies[0].left == seg_len
        and anomalies[0].right == 2 * seg_len - 1
    )


@pytest.mark.parametrize("Estimator", anomaly_detectors)
def test_anomaly_detector_dense(Estimator):
    """Tests dense anomaly detector output.

    Check if the predicted anomalies matches.
    """
    n_segments = 2
    seg_len = 50
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator(fmt="dense", labels="int_label")
    labels = detector.fit_predict(df)
    assert labels.nunique() == n_segments
    assert labels[seg_len - 1] == 0.0 and labels[seg_len] == 1.0
