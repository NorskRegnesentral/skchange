"""Basic tests for all anomaly detectors."""

import pandas as pd
import pytest
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.annotation import make_annotation_problem

from skchange.anomaly_detectors.anomalisers import StatThresholdAnomaliser
from skchange.anomaly_detectors.capa import Capa
from skchange.anomaly_detectors.circular_binseg import CircularBinarySegmentation
from skchange.anomaly_detectors.moscore_anomaly import MoscoreAnomaly
from skchange.anomaly_detectors.mvcapa import Mvcapa
from skchange.datasets.generate import generate_anomalous_data

anomaly_detectors = [
    Capa,
    CircularBinarySegmentation,
    MoscoreAnomaly,
    Mvcapa,
    StatThresholdAnomaliser,
]

true_anomalies = [(50, 59), (120, 129)]
anomaly_data = generate_anomalous_data(
    200, anomalies=true_anomalies, means=[10.0, 5.0], random_state=39
)


@pytest.mark.parametrize("Estimator", anomaly_detectors)
def test_output_type(Estimator):
    """Test annotator output type."""
    estimator = Estimator.create_test_instance()
    if not run_test_for_class(Estimator):
        return None

    arg = make_annotation_problem(
        n_timepoints=500, estimator_type=estimator.get_tag("distribution_type")
    )
    estimator.fit(arg)
    arg = make_annotation_problem(
        n_timepoints=200, estimator_type=estimator.get_tag("distribution_type")
    )
    y_pred = estimator.predict(arg)
    assert isinstance(y_pred, (pd.DataFrame, pd.Series))


@pytest.mark.parametrize("Estimator", anomaly_detectors)
def test_anomaly_detector_sparse_int(Estimator):
    """Test sparse int label anomaly detector output.

    Check if the predicted anomalies match.
    """
    detector = Estimator.create_test_instance()
    detector.set_params(fmt="sparse", labels="int_label")
    anomalies = detector.fit_predict(anomaly_data)
    assert len(anomalies) == len(true_anomalies)
    for i, (start, end) in enumerate(true_anomalies):
        assert anomalies.loc[i, "start"] == start and anomalies.loc[i, "end"] == end


@pytest.mark.parametrize("Estimator", anomaly_detectors)
def test_anomaly_detector_sparse_indicator(Estimator):
    """Test sparse indicator anomaly detector output.

    Check if the predicted anomalies match.
    """
    detector = Estimator.create_test_instance()
    detector.set_params(fmt="sparse", labels="indicator")
    anomalies = detector.fit_predict(anomaly_data)
    assert len(anomalies) == len(true_anomalies)
    for i, (start, end) in enumerate(true_anomalies):
        assert anomalies.loc[i, "start"] == start and anomalies.loc[i, "end"] == end


@pytest.mark.parametrize("Estimator", anomaly_detectors)
def test_anomaly_detector_score(Estimator):
    """Test score anomaly detector output."""
    sparse_detector = Estimator.create_test_instance()
    sparse_detector.set_params(fmt="sparse", labels="score")
    dense_detector = Estimator.create_test_instance()
    dense_detector.set_params(fmt="dense", labels="score")
    sparse_scores = sparse_detector.fit_predict(anomaly_data)
    dense_scores = dense_detector.fit_predict(anomaly_data)
    assert (sparse_scores == dense_scores).all(axis=None)
    if isinstance(sparse_scores, pd.DataFrame):
        assert "score" in sparse_scores.columns
    else:
        assert sparse_scores.name == "score"


@pytest.mark.parametrize("Estimator", anomaly_detectors)
def test_anomaly_detector_dense_int(Estimator):
    """Tests dense int label anomaly detector output.

    Check if the predicted anomalies matches.
    """
    detector = Estimator.create_test_instance()
    detector.set_params(fmt="dense", labels="int_label")
    labels = detector.fit_predict(anomaly_data)
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]

    assert labels.nunique() == len(true_anomalies) + 1
    for i, (start, end) in enumerate(true_anomalies):
        assert (labels.iloc[start : end + 1] == i + 1).all()


@pytest.mark.parametrize("Estimator", anomaly_detectors)
def test_anomaly_detector_dense_indicator(Estimator):
    """Tests dense indicator anomaly detector output.

    Check if the predicted anomalies matches.
    """
    detector = Estimator.create_test_instance()
    detector.set_params(fmt="dense", labels="indicator")
    labels = detector.fit_predict(anomaly_data)
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]

    for start, end in true_anomalies:
        assert labels.iloc[start : end + 1].all()
        assert not labels.iloc[start - 1] and not labels.iloc[end + 1]
