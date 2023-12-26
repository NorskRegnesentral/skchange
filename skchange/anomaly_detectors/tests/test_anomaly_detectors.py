"""Basic tests for all anomaly detectors."""

import numpy as np
import pandas as pd
import pytest
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.annotation import make_annotation_problem

from skchange.anomaly_detectors.capa import Capa
from skchange.anomaly_detectors.mvcapa import Mvcapa
from skchange.datasets.generate import teeth

anomaly_detectors = [Capa, Mvcapa]


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
    assert isinstance(y_pred, (pd.DataFrame, pd.Series, np.ndarray))


@pytest.mark.parametrize("Estimator", anomaly_detectors)
def test_anomaly_detector_sparse_int(Estimator):
    """Test sparse int label anomaly detector output.

    Check if the predicted anomalies match.
    """
    n_segments = 2
    seg_len = 20
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator(fmt="sparse", labels="int_label")
    anomalies = detector.fit_predict(df)
    assert (
        len(anomalies) == 1
        and anomalies.loc[0, "start"] == seg_len
        and anomalies.loc[0, "end"] == 2 * seg_len - 1
    )


@pytest.mark.parametrize("Estimator", anomaly_detectors)
def test_anomaly_detector_sparse_indicator(Estimator):
    """Test sparse indicator anomaly detector output.

    Check if the predicted anomalies match.
    """
    n_segments = 2
    seg_len = 20
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator(fmt="sparse", labels="indicator")
    anomalies = detector.fit_predict(df)
    assert (
        len(anomalies) == 1
        and anomalies.loc[0, "start"] == seg_len
        and anomalies.loc[0, "end"] == 2 * seg_len - 1
    )


@pytest.mark.parametrize("Estimator", anomaly_detectors)
def test_anomaly_detector_score(Estimator):
    """Test score anomaly detector output."""
    n_segments = 2
    seg_len = 20
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    sparse_detector = Estimator(fmt="sparse", labels="score")
    dense_detector = Estimator(fmt="dense", labels="score")
    sparse_scores = sparse_detector.fit_predict(df)
    dense_scores = dense_detector.fit_predict(df)
    assert sparse_scores.size == df.size
    assert (sparse_scores == dense_scores).all()


@pytest.mark.parametrize("Estimator", anomaly_detectors)
def test_anomaly_detector_dense_int(Estimator):
    """Tests dense int label anomaly detector output.

    Check if the predicted anomalies matches.
    """
    n_segments = 2
    seg_len = 50
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator(fmt="dense", labels="int_label")
    labels = detector.fit_predict(df)
    if isinstance(labels, pd.Series):
        assert labels.nunique() == n_segments
        assert labels.iloc[seg_len - 1] == 0.0 and labels.iloc[seg_len] == 1.0
    else:
        assert labels.nunique().iloc[0] == n_segments
        assert labels.iloc[seg_len - 1, 0] == 0.0 and labels.iloc[seg_len, 0] == 1.0


@pytest.mark.parametrize("Estimator", anomaly_detectors)
def test_anomaly_detector_dense_indicator(Estimator):
    """Tests dense indicator anomaly detector output.

    Check if the predicted anomalies matches.
    """
    n_segments = 2
    seg_len = 50
    df = teeth(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator(fmt="dense", labels="indicator")
    labels = detector.fit_predict(df)
    if isinstance(labels, pd.Series):
        assert not labels.iloc[seg_len - 1] and labels.iloc[seg_len]
    else:
        assert not labels.iloc[seg_len - 1, 0] and labels.iloc[seg_len, 0]
