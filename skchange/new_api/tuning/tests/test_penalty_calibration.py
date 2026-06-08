"""Tests for ``skchange.new_api.tuning._penalty_calibration``."""

import numpy as np
import pytest

from skchange.new_api.conftest import make_single_change_X
from skchange.new_api.detectors import SeededBinarySegmentation
from skchange.new_api.detectors.tests._registry import DETECTOR_TEST_INSTANCES
from skchange.new_api.interval_scorers import is_penalised_score
from skchange.new_api.interval_scorers._savings.multivariate_t_saving import (
    MultivariateTSaving,
)
from skchange.new_api.tuning import penalty_curve, unpenalised_scores

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


# ---------------------------------------------------------------------------
# penalty_param input-type contract
# ---------------------------------------------------------------------------


def _seeded_binseg_scores_for_param(penalty_param, *, no_penalty_value=0.0):
    """Run ``unpenalised_scores`` with a fresh SeededBinarySegmentation."""
    detector = SeededBinarySegmentation()
    X = make_single_change_X(detector)
    return unpenalised_scores(
        detector, X, penalty_param, no_penalty_value=no_penalty_value
    )


def test_unpenalised_scores_str_iterable_mapping_agree():
    """``penalty_param`` accepts ``str``, iterables of ``str`` and mappings,
    and the three forms produce identical scores when they encode the same
    parameter settings.
    """
    from_str = _seeded_binseg_scores_for_param("penalty")
    from_list = _seeded_binseg_scores_for_param(["penalty"])
    from_tuple = _seeded_binseg_scores_for_param(("penalty",))
    from_mapping = _seeded_binseg_scores_for_param({"penalty": 0.0})

    np.testing.assert_array_equal(from_str, from_list)
    np.testing.assert_array_equal(from_str, from_tuple)
    np.testing.assert_array_equal(from_str, from_mapping)


def test_unpenalised_scores_mapping_ignores_no_penalty_value():
    """When ``penalty_param`` is a mapping, ``no_penalty_value`` is ignored
    and per-key values are used directly.
    """
    explicit = _seeded_binseg_scores_for_param({"penalty": 0.0})
    # A nonsense ``no_penalty_value`` would change the result if it were used.
    overridden = _seeded_binseg_scores_for_param({"penalty": 0.0}, no_penalty_value=1e9)
    np.testing.assert_array_equal(explicit, overridden)


def test_unpenalised_scores_iterable_uses_no_penalty_value():
    """When ``penalty_param`` is an iterable, ``no_penalty_value`` is broadcast
    to every named parameter.
    """
    via_iterable = _seeded_binseg_scores_for_param(["penalty"], no_penalty_value=0.0)
    via_mapping = _seeded_binseg_scores_for_param({"penalty": 0.0})
    np.testing.assert_array_equal(via_iterable, via_mapping)


def test_unpenalised_scores_invalid_penalty_param_type_raises():
    """Non-(str / iterable / mapping) values for ``penalty_param`` are
    rejected by ``@validate_params``.
    """
    detector = SeededBinarySegmentation()
    X = make_single_change_X(detector)
    with pytest.raises((TypeError, ValueError)):
        unpenalised_scores(detector, X, penalty_param=123)


# ---------------------------------------------------------------------------
# penalty_curve
# ---------------------------------------------------------------------------


def _binseg_with_data():
    detector = SeededBinarySegmentation()
    X = make_single_change_X(detector)
    return detector, X


def test_penalty_curve_returns_array_aligned_with_penalty_range():
    """Output is a 1-D ndarray of the same length as ``penalty_range``."""
    detector, X = _binseg_with_data()
    penalty_range = np.array([1.0, 10.0, 100.0])

    curve = penalty_curve(
        detector, X, penalty_name="penalty", penalty_range=penalty_range
    )

    assert isinstance(curve, np.ndarray)
    assert curve.shape == penalty_range.shape


def test_penalty_curve_n_changepoints_decreases_with_penalty():
    """A very large penalty must yield fewer changepoints than a small one;
    confirms the sweep actually refits the detector at each value rather
    than reusing one fitted state.
    """
    detector, X = _binseg_with_data()
    penalty_range = np.array([1e-3, 1e9])

    curve = penalty_curve(
        detector,
        X,
        penalty_name="penalty",
        penalty_range=penalty_range,
        scoring="n_changepoints",
    )

    assert curve[-1] <= curve[0]
    assert curve[-1] == 0.0


def test_penalty_curve_accepts_callable_scoring():
    """A user-supplied scorer callable is invoked once per candidate and its
    return value populates the curve.
    """
    detector, X = _binseg_with_data()
    penalty_range = np.array([1.0, 10.0, 100.0])
    n_calls = 0

    def constant_scorer(detector, X, y=None):
        nonlocal n_calls
        n_calls += 1
        return 42.0

    curve = penalty_curve(
        detector,
        X,
        penalty_name="penalty",
        penalty_range=penalty_range,
        scoring=constant_scorer,
    )

    np.testing.assert_array_equal(curve, np.full(penalty_range.shape, 42.0))
    assert n_calls == penalty_range.size


def test_penalty_curve_forwards_y_to_scorer_only():
    """``y`` is passed through to the scorer but never to ``detector.fit``."""
    detector, X = _binseg_with_data()
    penalty_range = np.array([1.0, 10.0])
    sentinel_y = np.array([0, 1, 2])
    seen_ys = []

    def y_capturing_scorer(detector, X, y=None):
        seen_ys.append(y)
        return 0.0

    penalty_curve(
        detector,
        X,
        sentinel_y,
        penalty_name="penalty",
        penalty_range=penalty_range,
        scoring=y_capturing_scorer,
    )

    assert len(seen_ys) == penalty_range.size
    for y in seen_ys:
        assert y is sentinel_y


def test_penalty_curve_does_not_mutate_input_detector():
    """The input detector's parameters are not modified by the sweep
    (sklearn-style: ``clone`` per iteration).
    """
    detector, X = _binseg_with_data()
    original_penalty = detector.get_params()["penalty"]
    penalty_range = np.array([1.0, 10.0, 100.0])

    penalty_curve(detector, X, penalty_name="penalty", penalty_range=penalty_range)

    assert detector.get_params()["penalty"] == original_penalty


@pytest.mark.parametrize(
    "bad_range",
    [
        [],  # empty
        [[1.0, 2.0], [3.0, 4.0]],  # 2-D
        ["a", "b"],  # non-numeric strings
        [None, 1.0],  # None mixed in
    ],
)
def test_penalty_curve_invalid_penalty_range_raises(bad_range):
    detector, X = _binseg_with_data()
    with pytest.raises((TypeError, ValueError)):
        penalty_curve(
            detector,
            X,
            penalty_name="penalty",
            penalty_range=bad_range,
        )
