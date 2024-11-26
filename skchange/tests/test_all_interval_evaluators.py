import numpy as np
import pytest
from sktime.utils._testing.annotation import make_annotation_problem
from sktime.utils.estimator_checks import check_estimator, parametrize_with_checks

from skchange.anomaly_scores import ANOMALY_SCORES
from skchange.change_scores import CHANGE_SCORES
from skchange.costs import COSTS
from skchange.datasets import generate_alternating_data

INTERVAL_EVALUATORS = COSTS + CHANGE_SCORES + ANOMALY_SCORES


@parametrize_with_checks(INTERVAL_EVALUATORS)
def test_sktime_compatible_estimators(obj, test_name):
    check_estimator(obj, tests_to_run=test_name, raise_exceptions=True)


@pytest.mark.parametrize("Evaluator", INTERVAL_EVALUATORS)
def test_evaluator_fit(Evaluator):
    evaluator = Evaluator.create_test_instance()
    x = make_annotation_problem(n_timepoints=50, estimator_type="None")
    fit_evaluator = evaluator.fit(x)
    assert fit_evaluator._is_fitted


@pytest.mark.parametrize("Evaluator", INTERVAL_EVALUATORS)
def test_evaluator_evaluate(Evaluator):
    evaluator = Evaluator.create_test_instance()
    x = make_annotation_problem(n_timepoints=50, estimator_type="None")
    evaluator.fit(x)
    cut1 = np.linspace(0, 10, evaluator.expected_cut_entries, dtype=int)

    results = evaluator.evaluate(cut1)
    assert isinstance(results, np.ndarray)
    assert results.ndim == 2
    assert len(results) == 1

    cut2 = np.linspace(10, 20, evaluator.expected_cut_entries, dtype=int)
    cuts = np.array([cut1, cut2])
    results = evaluator.evaluate(cuts)
    assert isinstance(results, np.ndarray)
    assert results.ndim == 2
    assert len(results) == len(cuts)


@pytest.mark.parametrize("Evaluator", INTERVAL_EVALUATORS)
def test_evaluator_evaluate_by_evaluation_type(Evaluator):
    evaluator = Evaluator.create_test_instance()
    n_segments = 1
    seg_len = 50
    p = 3
    df = generate_alternating_data(
        n_segments=n_segments,
        mean=20,
        segment_length=seg_len,
        p=p,
        random_state=15,
    )

    evaluator.fit(df)
    cut1 = np.linspace(0, 10, evaluator.expected_cut_entries, dtype=int)
    cut2 = np.linspace(10, 20, evaluator.expected_cut_entries, dtype=int)
    cuts = np.array([cut1, cut2])

    results = evaluator.evaluate(cuts)

    if evaluator.evaluation_type == "univariate":
        assert results.shape == (2, p)
    elif evaluator.evaluation_type == "multivariate":
        assert results.shape == (2, 1)
    else:
        raise ValueError("Invalid scitype:evaluator tag.")


@pytest.mark.parametrize("Evaluator", INTERVAL_EVALUATORS)
def test_evaluator_invalid_cuts(Evaluator):
    evaluator = Evaluator.create_test_instance()
    x = make_annotation_problem(n_timepoints=50, estimator_type="None")
    evaluator.fit(x)
    with pytest.raises(ValueError):
        cut = np.linspace(0, 10, evaluator.expected_cut_entries, dtype=float)
        evaluator.evaluate(cut)
    with pytest.raises(ValueError):
        cut = np.linspace(0, 10, evaluator.expected_cut_entries - 1, dtype=int)
        evaluator.evaluate(cut)
    with pytest.raises(ValueError):
        cut = np.linspace(0, 10, evaluator.expected_cut_entries + 1, dtype=int)
        evaluator.evaluate(cut)
    with pytest.raises(ValueError):
        cut = np.linspace(10, 0, evaluator.expected_cut_entries, dtype=int)
        evaluator.evaluate(cut)
