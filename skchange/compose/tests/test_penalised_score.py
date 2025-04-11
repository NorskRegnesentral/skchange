"""Tests for penalised scores."""

import numpy as np
import pandas as pd
import pytest

from skchange.change_scores import CUSUM, MultivariateGaussianScore
from skchange.compose.penalised_score import PenalisedScore
from skchange.penalties import BICPenalty, LinearChiSquarePenalty


def test_penalised_score_init():
    scorer = CUSUM()
    penalty = BICPenalty()
    penalised_score = PenalisedScore(scorer, penalty)
    assert penalised_score.expected_cut_entries == scorer.expected_cut_entries

    with pytest.raises(ValueError, match="penalised"):
        PenalisedScore(PenalisedScore(scorer, penalty), penalty)

    scorer = MultivariateGaussianScore()
    penalty = LinearChiSquarePenalty()

    with pytest.raises(ValueError):
        PenalisedScore(scorer, penalty)


def test_penalised_score_fit():
    scorer = CUSUM()
    penalty = BICPenalty()

    df2 = pd.DataFrame(np.random.randn(100, 2))
    df3 = pd.DataFrame(np.random.randn(100, 3))

    penalty.fit(df3, scorer)
    penalised_score = PenalisedScore(scorer, penalty)
    penalised_score.fit(df3, scorer)
    assert penalised_score.is_fitted

    penalty.fit(df2, scorer)
    penalised_score = PenalisedScore(scorer, penalty)

    with pytest.raises(ValueError):
        penalised_score.fit(df3, scorer)


def test_penalised_score_get_param_size():
    scorer = CUSUM()
    penalty = BICPenalty()
    penalised_score = PenalisedScore(scorer, penalty)
    assert penalised_score.get_param_size(1) == scorer.get_param_size(1)
    assert penalised_score.get_param_size(5) == scorer.get_param_size(5)
