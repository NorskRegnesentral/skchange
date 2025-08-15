"""Comprehensive tests for all interval scorers in skchange.

This module implements pytest best practices including:
- Fixtures for common test data
- Proper test organization
- Descriptive test names
- Comprehensive error testing
- Performance considerations
"""

import numpy as np
import pandas as pd
import pytest
from sktime.tests.test_all_estimators import VALID_ESTIMATOR_TAGS

from skchange.anomaly_scores import ANOMALY_SCORES
from skchange.base import BaseIntervalScorer
from skchange.change_scores import CHANGE_SCORES
from skchange.compose.penalised_score import PenalisedScore
from skchange.costs import COSTS
from skchange.datasets import generate_alternating_data, generate_anomalous_data

INTERVAL_SCORERS = COSTS + CHANGE_SCORES + ANOMALY_SCORES + [PenalisedScore]
VALID_SCORER_TAGS = list(VALID_ESTIMATOR_TAGS) + [
    "task",
    "distribution_type",
    "is_conditional",
    "is_aggregated",
    "is_penalised",
    "supports_fixed_param",
]


def skip_if_no_test_data(scorer: BaseIntervalScorer) -> None:
    """Skip test if scorer doesn't have appropriate test data.

    Args:
        scorer: The scorer instance to check.

    Raises:
        pytest.skip: If scorer doesn't have test data available.
    """
    distribution_type = scorer.get_tag("distribution_type")
    is_conditional = scorer.get_tag("is_conditional")
    if distribution_type == "Poisson" or is_conditional:
        pytest.skip(
            f"{scorer.__class__.__name__} does not have test data in place yet."
        )


@pytest.mark.parametrize("Scorer", INTERVAL_SCORERS)
def test_task_tag_set(Scorer: type[BaseIntervalScorer]) -> None:
    """Test that each scorer has a valid task tag set.

    Args:
        Scorer: The scorer class to test.
    """
    scorer = Scorer.create_test_instance()
    valid_tasks = ["cost", "change_score", "saving", "local_anomaly_score"]
    task = scorer.get_tag("task")

    assert task is not None, f"{Scorer.__name__} has no task tag set"
    assert (
        task in valid_tasks
    ), f"{Scorer.__name__} has invalid task '{task}'. Must be one of {valid_tasks}"


@pytest.mark.parametrize("Scorer", INTERVAL_SCORERS)
def test_scorer_fit(
    Scorer: type[BaseIntervalScorer],
    sample_anomalous_data: pd.DataFrame,
) -> None:
    """Test that scorers can be fitted successfully.

    Args:
        Scorer: The scorer class to test.
        sample_anomalous_data: Sample data fixture for fitting.
    """
    scorer = Scorer.create_test_instance()
    skip_if_no_test_data(scorer)

    # Test fitting
    fit_scorer = scorer.fit(sample_anomalous_data)

    # Verify fit was successful
    assert fit_scorer.is_fitted, f"{Scorer.__name__} not marked as fitted after fit()"
    assert fit_scorer is scorer, f"{Scorer.__name__} fit() should return self"


@pytest.mark.parametrize("Scorer", INTERVAL_SCORERS)
def test_scorer_evaluate(
    Scorer: type[BaseIntervalScorer],
    sample_anomalous_data: pd.DataFrame,
) -> None:
    """Test scorer evaluation with single and multiple cuts.

    Args:
        Scorer: The scorer class to test.
        sample_anomalous_data: Sample data fixture for evaluation.
    """
    scorer = Scorer.create_test_instance()
    skip_if_no_test_data(scorer)

    scorer.fit(sample_anomalous_data)

    # Test single cut evaluation
    cut1 = np.linspace(0, 40, scorer._get_required_cut_size(), dtype=int)
    results = scorer.evaluate(cut1)

    assert isinstance(
        results, np.ndarray
    ), f"{Scorer.__name__} evaluate() should return numpy array"
    assert (
        results.ndim == 2
    ), f"{Scorer.__name__} evaluate() should return 2D array, got {results.ndim}D"
    assert (
        len(results) == 1
    ), f"{Scorer.__name__} evaluate() should return array with 1 row for single cut"

    # Test multiple cuts evaluation
    cut2 = np.linspace(10, 40, scorer._get_required_cut_size(), dtype=int)
    cuts = np.array([cut1, cut2])
    results = scorer.evaluate(cuts)

    assert isinstance(
        results, np.ndarray
    ), f"{Scorer.__name__} evaluate() should return numpy array for multiple cuts"
    assert (
        results.ndim == 2
    ), f"{Scorer.__name__} evaluate() should return 2D array for multiple cuts"
    assert len(results) == len(
        cuts
    ), f"{Scorer.__name__} evaluate() should return one row per cut"


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
    cut1 = np.linspace(0, 20, scorer._get_required_cut_size(), dtype=int)
    cut2 = np.linspace(20, 40, scorer._get_required_cut_size(), dtype=int)
    cuts = np.array([cut1, cut2])

    results = scorer.evaluate(cuts)

    is_aggregated = scorer.get_tag("is_aggregated")
    is_conditional = scorer.get_tag("is_conditional")
    if is_aggregated:
        assert results.shape == (2, 1)
    elif not is_aggregated and is_conditional:
        assert results.shape[0] == 2
        assert results.shape[1] >= 1 and results.shape[1] <= p - 1
    else:
        assert results.shape == (2, p)


@pytest.mark.parametrize("Scorer", INTERVAL_SCORERS)
class TestScorerInvalidCuts:
    """Test class for invalid cut scenarios."""

    @pytest.fixture(autouse=True)
    def setup_scorer(
        self,
        Scorer: type[BaseIntervalScorer],
        sample_anomalous_data: pd.DataFrame,
    ):
        """Set up fitted scorer for all tests in this class."""
        self.scorer = Scorer.create_test_instance()
        skip_if_no_test_data(self.scorer)
        self.scorer.fit(sample_anomalous_data)

    def test_float_cuts_raise_error(self):
        """Test that float cuts raise ValueError."""
        cut = np.linspace(0, 10, self.scorer._get_required_cut_size(), dtype=float)
        with pytest.raises(ValueError, match="must be integers"):
            self.scorer.evaluate(cut)

    def test_wrong_cut_size_smaller_raises_error(self):
        """Test that cuts with insufficient size raise ValueError."""
        cut = np.linspace(0, 10, self.scorer._get_required_cut_size() - 1, dtype=int)
        with pytest.raises(ValueError, match="size"):
            self.scorer.evaluate(cut)

    def test_wrong_cut_size_larger_raises_error(self):
        """Test that cuts with excessive size raise ValueError."""
        cut = np.linspace(0, 10, self.scorer._get_required_cut_size() + 1, dtype=int)
        with pytest.raises(ValueError, match="size"):
            self.scorer.evaluate(cut)

    def test_reversed_cuts_raise_error(self):
        """Test that reversed cuts (end < start) raise ValueError."""
        cut = np.linspace(10, 0, self.scorer._get_required_cut_size(), dtype=int)
        error_pattern = "start.*end|end.*start|decreasing|increasing"
        with pytest.raises(ValueError, match=error_pattern):
            self.scorer.evaluate(cut)


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
    assert scorer.get_model_size(1) >= 0

    skip_if_no_test_data(scorer)
    x = generate_anomalous_data()
    scorer.fit(x)
    assert scorer.get_model_size(1) >= 0


@pytest.mark.parametrize("Scorer", INTERVAL_SCORERS)
def test_valid_interval_scorer_class_tags(Scorer: type[BaseIntervalScorer]):
    """Check that Scorer class tags are in VALID_SCORER_TAGS."""
    for tag in Scorer.get_class_tags().keys():
        msg = "Found invalid tag: %s" % tag
        assert tag in VALID_SCORER_TAGS, msg


@pytest.mark.parametrize("Scorer", INTERVAL_SCORERS)
def test_valid_interval_scorer_tags(Scorer: type[BaseIntervalScorer]):
    """Check that Scorer tags are in VALID_SCORER_TAGS."""
    scorer = Scorer.create_test_instance()
    for tag in scorer.get_tags().keys():
        msg = "Found invalid tag: %s" % tag
        assert tag in VALID_SCORER_TAGS, msg
