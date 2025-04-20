"""Tests for interval scorer validation."""

import pytest

from skchange.costs import L2Cost, LinearRegressionCost, MultivariateGaussianCost
from skchange.utils.validation.interval_scorer import check_interval_scorer


def test_check_interval_scorer_univariate():
    """Test check_interval_scorer with univariate evaluation type."""
    scorer = L2Cost()
    check_interval_scorer(
        scorer,
        arg_name="scorer",
        caller_name="test_check_interval_scorer_univariate",
        require_evaluation_type="univariate",
    )


def test_check_interval_scorer_multivariate():
    """Test check_interval_scorer with multivariate evaluation type."""
    scorer = MultivariateGaussianCost()
    check_interval_scorer(
        scorer,
        arg_name="scorer",
        caller_name="test_check_interval_scorer_multivariate",
        require_evaluation_type="multivariate",
    )


def test_check_interval_scorer_conditional():
    """Test check_interval_scorer with conditional evaluation type."""
    scorer = LinearRegressionCost.create_test_instance()
    check_interval_scorer(
        scorer,
        arg_name="scorer",
        caller_name="test_check_interval_scorer_conditional",
        require_evaluation_type="conditional",
    )


def test_check_interval_scorer_invalid_type():
    """Test check_interval_scorer raises error for invalid evaluation type."""
    scorer = L2Cost()
    with pytest.raises(ValueError, match="requires `scorer` to have multivariate"):
        check_interval_scorer(
            scorer,
            arg_name="scorer",
            caller_name="test_check_interval_scorer_invalid_type",
            require_evaluation_type="multivariate",
        )
