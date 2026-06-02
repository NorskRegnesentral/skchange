"""Tests for ``skchange.new_api.tuning.unpenalised_scores``."""

import numpy as np
import pytest

from skchange.new_api.conftest import make_single_change_X
from skchange.new_api.detectors.tests._registry import DETECTOR_TEST_INSTANCES
from skchange.new_api.interval_scorers import is_penalised_score
from skchange.new_api.interval_scorers._savings.multivariate_t_saving import (
    MultivariateTSaving,
)
from skchange.new_api.tuning import unpenalised_scores

_all_detectors = pytest.mark.parametrize(
    "estimator", DETECTOR_TEST_INSTANCES, indirect=True, ids=repr
)

# Detector parameter names that hold the underlying interval scorer, in the
# order they are tried. The first one that resolves to a non-None scorer wins.
_SCORER_PARAM_NAMES = (
    "change_score",
    "transient_score",
    "segment_saving",
    "cost",
)


def _underlying_scorer(estimator):
    """Return the underlying interval scorer if exposed via a known param name."""
    params = estimator.get_params(deep=False)
    for name in _SCORER_PARAM_NAMES:
        if name in params and params[name] is not None:
            return params[name]
    return None


@_all_detectors
def test_unpenalised_scores_sanity(estimator):
    """`unpenalised_scores` must return finite values, and non-negative (up to
    floating-point noise) when the underlying scorer declares the
    ``non_negative_scores`` tag.

    Cost-based detectors (e.g. PELT) and inherently penalised scorers (e.g.
    ``ESACScore``) are only checked for finiteness.
    """
    if not hasattr(estimator, "predict_scores"):
        pytest.skip("predict_scores not implemented")
    params = estimator.get_params(deep=False)
    if "penalty_scale" not in params:
        pytest.skip("detector has no top-level 'penalty_scale' parameter")

    X = make_single_change_X(estimator)
    scores = unpenalised_scores(estimator, X, "penalty_scale")

    assert isinstance(scores, np.ndarray)
    assert scores.ndim == 1
    assert np.all(np.isfinite(scores)), "unpenalised scores must be finite"

    scorer = _underlying_scorer(estimator)
    if scorer is None:
        return
    if is_penalised_score(scorer):
        return
    if isinstance(scorer, MultivariateTSaving):
        # The multivariate-T MLE is iterative and not exactly subadditive in
        # finite samples, so the raw saving can take meaningfully negative
        # values from numerical noise. Skipped from the non-negativity check.
        pytest.skip("MultivariateTSaving is not strictly non-negative in practice")

    tol = 1e-10
    assert np.all(scores >= -tol), (
        "unpenalised scores from a scorer tagged `non_negative_scores=True` "
        f"must be non-negative (tol={tol}), got min={scores.min()}"
    )
