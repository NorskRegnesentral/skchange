import numpy as np
import pandas as pd
import pytest

from skchange.anomaly_scores import ANOMALY_SCORES
from skchange.base import BaseIntervalScorer
from skchange.change_scores import CHANGE_SCORES
from skchange.compose.penalised_score import PenalisedScore
from skchange.costs import COSTS, LinearRegressionCost
from skchange.datasets import generate_alternating_data, generate_anomalous_data
from skchange.utils.validation.enums import EvaluationType

SIMPLE_EVAL_TEST_EVALUATORS = COSTS + CHANGE_SCORES + ANOMALY_SCORES + [PenalisedScore]


@pytest.mark.parametrize("Evaluator", SIMPLE_EVAL_TEST_EVALUATORS)
def test_evaluator_fit(Evaluator: type[BaseIntervalScorer]):
    evaluator = Evaluator.create_test_instance()
    x = generate_anomalous_data()
    x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")
    fit_evaluator = evaluator.fit(x)
    assert fit_evaluator.is_fitted


@pytest.mark.parametrize("Evaluator", SIMPLE_EVAL_TEST_EVALUATORS)
def test_evaluator_evaluate(Evaluator: type[BaseIntervalScorer]):
    evaluator = Evaluator.create_test_instance()
    x = generate_anomalous_data()
    x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")
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


@pytest.mark.parametrize("Evaluator", SIMPLE_EVAL_TEST_EVALUATORS)
def test_evaluator_evaluate_by_evaluation_type(Evaluator: BaseIntervalScorer):
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

    if evaluator.evaluation_type == EvaluationType.UNIVARIATE:
        assert results.shape == (2, p)
        assert evaluator.output_dim == p
    elif evaluator.evaluation_type == EvaluationType.MULTIVARIATE:
        assert results.shape == (2, 1)
        assert evaluator.output_dim == 1
    else:
        raise ValueError("Invalid scitype:evaluator tag.")


def test_conditional_evaluator_output_dim():
    evaluator = LinearRegressionCost(response_col=0)
    X = np.random.randn(50, 3)
    evaluator.fit(X)
    assert evaluator.evaluation_type == EvaluationType.CONDITIONAL
    assert evaluator.output_dim == 1


@pytest.mark.parametrize("Evaluator", SIMPLE_EVAL_TEST_EVALUATORS)
def test_evaluator_invalid_cuts(Evaluator: BaseIntervalScorer):
    evaluator = Evaluator.create_test_instance()
    x = generate_anomalous_data()
    x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")
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
