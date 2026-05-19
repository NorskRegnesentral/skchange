"""Tests for the refactored ESACScore (Phase 0)."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from skchange.new_api.interval_scorers import ESACScore, PenalisedScore

_RNG = np.random.default_rng(0)
_N, _P = 100, 10
_X_NULL = _RNG.normal(size=(_N, _P))

_X_NULL_1D = _RNG.normal(size=(_N, 1))


# ---------------------------------------------------------------------------
# API contract: no constructor parameters
# ---------------------------------------------------------------------------


def test_esac_score_no_constructor_parameters():
    """ESACScore must have no constructor parameters after refactor."""
    scorer = ESACScore()
    assert scorer.get_params(deep=False) == {}


# ---------------------------------------------------------------------------
# Tag contract
# ---------------------------------------------------------------------------


def test_esac_score_penalised_tag_is_false():
    """ESACScore must report penalised=False so PenalisedScore can wrap it."""
    tags = ESACScore().__sklearn_tags__().interval_scorer_tags
    assert tags.penalised is False


def test_esac_score_aggregated_tag_is_true():
    """ESACScore must report aggregated=True."""
    tags = ESACScore().__sklearn_tags__().interval_scorer_tags
    assert tags.aggregated is True


def test_esac_score_non_negative_scores_tag_is_false():
    """ESACScore scores can be negative (ratio can be below 1)."""
    tags = ESACScore().__sklearn_tags__().interval_scorer_tags
    assert tags.non_negative_scores is False


# ---------------------------------------------------------------------------
# get_default_penalty
# ---------------------------------------------------------------------------


def test_esac_score_get_default_penalty_before_fit_raises():
    """get_default_penalty() must raise NotFittedError before fit."""
    with pytest.raises(NotFittedError):
        ESACScore().get_default_penalty()


def test_esac_score_get_default_penalty_returns_1():
    """After fit, get_default_penalty() must return exactly 1.0."""
    scorer = ESACScore().fit(_X_NULL)
    assert scorer.get_default_penalty() == 1.0


def test_esac_score_get_default_penalty_returns_1_univariate():
    """get_default_penalty() returns 1.0 for p=1 too."""
    scorer = ESACScore().fit(_X_NULL_1D)
    assert scorer.get_default_penalty() == 1.0


# ---------------------------------------------------------------------------
# PenalisedScore wrapping
# ---------------------------------------------------------------------------


def test_penalised_score_wrapping_esac_does_not_raise():
    """PenalisedScore(ESACScore()).fit(X) must NOT raise after Phase 0 refactor."""
    ps = PenalisedScore(ESACScore())
    ps.fit(_X_NULL)  # must not raise


def test_penalised_score_wrapping_esac_evaluate_shape():
    """PenalisedScore(ESACScore()).evaluate() must return shape (n_specs, 1)."""
    ps = PenalisedScore(ESACScore(), penalty=1.0).fit(_X_NULL)
    cache = ps.precompute(_X_NULL)
    n_specs = 5
    specs = np.array(
        [[i * 10, i * 10 + 5, i * 10 + 10] for i in range(n_specs)], dtype=np.int64
    )
    out = ps.evaluate(cache, specs)
    assert out.shape == (n_specs, 1)


# ---------------------------------------------------------------------------
# Null data: score should be ≤ 0 most of the time at C=1
# ---------------------------------------------------------------------------


def test_penalised_esac_mostly_nonpositive_on_null_data():
    """At penalty=1.0, >99% of splits on i.i.d. Gaussian data should be ≤ 0."""
    ps = PenalisedScore(ESACScore(), penalty=1.0).fit(_X_NULL)
    cache = ps.precompute(_X_NULL)
    n = _N
    specs = np.array(
        [[0, t, n] for t in range(1, n)],
        dtype=np.int64,
    )
    scores = ps.evaluate(cache, specs).reshape(-1)
    frac_positive = np.mean(scores > 0)
    assert frac_positive < 0.01, (
        f"Expected <1% positive scores on null data at C=1, got {frac_positive:.1%}"
    )
