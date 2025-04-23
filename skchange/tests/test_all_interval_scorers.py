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

INTERVAL_SCORERS = COSTS + CHANGE_SCORES + ANOMALY_SCORES + [PenalisedScore]


def skip_if_no_test_data(scorer: BaseIntervalScorer):
    if (
        scorer.get_tag("distribution_type") == "Poisson"
        or scorer.evaluation_type == EvaluationType.CONDITIONAL
    ):
        pytest.skip(
            f"{scorer.__class__.__name__} does not have test data in place yet."
        )


@pytest.mark.parametrize("Scorer", INTERVAL_SCORERS)
def test_task_tag_set(Scorer: type[BaseIntervalScorer]):
    scorer = Scorer.create_test_instance()
    valid_tasks = ["cost", "change_score", "saving", "local_anomaly_score"]
    assert scorer.get_tag("task") in valid_tasks


@pytest.mark.parametrize("Scorer", INTERVAL_SCORERS)
def test_scorer_fit(Scorer: type[BaseIntervalScorer]):
    scorer = Scorer.create_test_instance()
    skip_if_no_test_data(scorer)
    x = generate_anomalous_data()
    x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")
    fit_scorer = scorer.fit(x)
    assert fit_scorer.is_fitted


@pytest.mark.parametrize("Scorer", INTERVAL_SCORERS)
def test_scorer_evaluate(Scorer: type[BaseIntervalScorer]):
    scorer = Scorer.create_test_instance()
    skip_if_no_test_data(scorer)
    x = generate_anomalous_data()
    x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")
    scorer.fit(x)
    cut1 = np.linspace(0, 10, scorer._get_required_cut_size(), dtype=int)

    results = scorer.evaluate(cut1)
    assert isinstance(results, np.ndarray)
    assert results.ndim == 2
    assert len(results) == 1

    cut2 = np.linspace(10, 20, scorer._get_required_cut_size(), dtype=int)
    cuts = np.array([cut1, cut2])
    results = scorer.evaluate(cuts)
    assert isinstance(results, np.ndarray)
    assert results.ndim == 2
    assert len(results) == len(cuts)


@pytest.mark.parametrize("Scorer", INTERVAL_SCORERS)
def test_scorer_evaluate_by_evaluation_type(Scorer: BaseIntervalScorer):
    scorer = Scorer.create_test_instance()
    skip_if_no_test_data(scorer)
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

    scorer.fit(df)
    cut1 = np.linspace(0, 10, scorer._get_required_cut_size(), dtype=int)
    cut2 = np.linspace(10, 20, scorer._get_required_cut_size(), dtype=int)
    cuts = np.array([cut1, cut2])

    results = scorer.evaluate(cuts)

    if scorer.evaluation_type == EvaluationType.UNIVARIATE:
        assert results.shape == (2, p)
        assert scorer.output_dim == p
    elif scorer.evaluation_type == EvaluationType.MULTIVARIATE:
        assert results.shape == (2, 1)
        assert scorer.output_dim == 1
    else:
        raise ValueError("Invalid scitype:scorer tag.")


def test_conditional_scorer_output_dim():
    scorer = LinearRegressionCost(response_col=0)
    X = np.random.randn(50, 3)
    scorer.fit(X)
    assert scorer.evaluation_type == EvaluationType.CONDITIONAL
    assert scorer.output_dim == 1


@pytest.mark.parametrize("Scorer", INTERVAL_SCORERS)
def test_scorer_invalid_cuts(Scorer: BaseIntervalScorer):
    scorer = Scorer.create_test_instance()
    skip_if_no_test_data(scorer)
    x = generate_anomalous_data()
    x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")
    scorer.fit(x)
    with pytest.raises(ValueError):
        cut = np.linspace(0, 10, scorer._get_required_cut_size(), dtype=float)
        scorer.evaluate(cut)
    with pytest.raises(ValueError):
        cut = np.linspace(0, 10, scorer._get_required_cut_size() - 1, dtype=int)
        scorer.evaluate(cut)
    with pytest.raises(ValueError):
        cut = np.linspace(0, 10, scorer._get_required_cut_size() + 1, dtype=int)
        scorer.evaluate(cut)
    with pytest.raises(ValueError):
        cut = np.linspace(10, 0, scorer._get_required_cut_size(), dtype=int)
        scorer.evaluate(cut)


@pytest.mark.parametrize("Scorer", INTERVAL_SCORERS)
def test_scorer_min_size(Scorer: BaseIntervalScorer):
    scorer = Scorer.create_test_instance()
    assert scorer.min_size is None or scorer.min_size >= 1

    skip_if_no_test_data(scorer)
    x = generate_anomalous_data()
    scorer.fit(x)
    assert scorer.min_size >= 1


@pytest.mark.parametrize("Scorer", INTERVAL_SCORERS)
def test_scorer_param_size(Scorer: BaseIntervalScorer):
    scorer = Scorer.create_test_instance()
    assert scorer.get_param_size(1) >= 0

    skip_if_no_test_data(scorer)
    x = generate_anomalous_data()
    scorer.fit(x)
    assert scorer.get_param_size(1) >= 0
