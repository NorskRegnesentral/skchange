"""Change scores as interval evaluators."""

from skchange.change_scores._continuous_linear_trend_score import (
    ContinuousLinearTrendScore,
)
from skchange.change_scores._cusum import CUSUM
from skchange.change_scores._from_cost import ChangeScore, to_change_score
from skchange.change_scores._linear_trend_score import BestFitLinearTrendScore
from skchange.change_scores._multivariate_gaussian_score import (
    MultivariateGaussianScore,
)

CHANGE_SCORES = [
    BestFitLinearTrendScore,
    ChangeScore,
    ContinuousLinearTrendScore,
    MultivariateGaussianScore,
    CUSUM,
]

__all__ = CHANGE_SCORES + [to_change_score]
