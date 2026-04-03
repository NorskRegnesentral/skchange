"""Shared pytest fixtures and test data helpers for ``skchange.new_api`` tests."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone

_N_SAMPLES = 100
CHANGEPOINT = 50


def make_single_change_X(estimator: BaseEstimator) -> np.ndarray:
    """Generate test data appropriate for the given estimator based on its tags.

    Parameters
    ----------
    estimator : BaseEstimator
        An estimator instance whose tags determine the generated data shape and type.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Two-segment data with a structural break at sample ``CHANGEPOINT``.
        Data type and number of features are chosen based on estimator tags:

        - ``input_tags.integer_only=True``: Poisson count data with a rate shift.
        - ``input_tags.integer_only=False``: Gaussian data with a mean shift.
        - ``interval_scorer_tags.conditional=True``: at least 2 features generated.
    """
    tags = estimator.__sklearn_tags__()
    input_tags = tags.input_tags
    interval_scorer_tags = getattr(tags, "interval_scorer_tags", None)

    n_features = 2 if (interval_scorer_tags and interval_scorer_tags.conditional) else 1

    rng = np.random.default_rng(51)
    if input_tags.integer_only:
        X = np.concatenate(
            [
                rng.poisson(lam=2.0, size=(CHANGEPOINT, n_features)),
                rng.poisson(lam=10.0, size=(_N_SAMPLES - CHANGEPOINT, n_features)),
            ]
        ).astype(float)
    else:
        X = np.concatenate(
            [
                rng.normal(loc=0.0, scale=1.0, size=(CHANGEPOINT, n_features)),
                rng.normal(
                    loc=10.0, scale=1.0, size=(_N_SAMPLES - CHANGEPOINT, n_features)
                ),
            ]
        )

    return X


def make_no_change_X(estimator: BaseEstimator) -> np.ndarray:
    """Generate stationary test data (no structural break) for the given estimator.

    Parameters
    ----------
    estimator : BaseEstimator
        An estimator instance whose tags determine the generated data shape and type.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Homogeneous data drawn from a single distribution throughout.
        Data type and number of features are chosen based on estimator tags:

        - ``input_tags.integer_only=True``: Poisson count data with constant rate.
        - ``input_tags.integer_only=False``: Gaussian data with constant mean.
        - ``interval_scorer_tags.conditional=True``: at least 2 features generated.
    """
    tags = estimator.__sklearn_tags__()
    input_tags = tags.input_tags
    interval_scorer_tags = getattr(tags, "interval_scorer_tags", None)

    n_features = 2 if (interval_scorer_tags and interval_scorer_tags.conditional) else 1

    rng = np.random.default_rng(42)
    if input_tags.integer_only:
        X = rng.poisson(lam=5.0, size=(_N_SAMPLES, n_features)).astype(float)
    else:
        X = rng.normal(loc=0.0, scale=1.0, size=(_N_SAMPLES, n_features))

    return X


@pytest.fixture
def estimator(request):
    """Clone a fresh unfitted estimator from a registry instance."""
    return clone(request.param)
