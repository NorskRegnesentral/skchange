"""Classes to construct other types of interval scorers from costs."""

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import (
    BaseChangeScore,
    BaseIntervalScorer,
)
from skchange.new_api.typing import ArrayLike, Self
from skchange.new_api.utils import SkchangeTags
from skchange.new_api.utils.validation import (
    check_interval_scorer,
    check_interval_specs,
    validate_data,
)


class CostChangeScore(BaseChangeScore):
    """Change scorer constructed from a cost scorer.

    Computes change score as the cost reduction of allowing a split within an interval:

    ``score = cost(start, end) - cost(start, split) - cost(split, end)``
    """

    def __init__(self, cost: BaseIntervalScorer):
        self.cost = cost

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit wrapped cost scorer."""
        X = validate_data(self, X, ensure_2d=True, reset=True)
        check_interval_scorer(
            self.cost,
            ensure_score_type=["cost"],
            caller_name=self.__class__.__name__,
            arg_name="cost",
        )
        self.cost_: BaseIntervalScorer = clone(self.cost).fit(X, y)
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute wrapped cost scorer data."""
        check_is_fitted(self, ["cost_"])
        return self.cost_.precompute(X)

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate change score on interval specifications.

        Parameters
        ----------
        cache : dict
            Cache from precompute().
        interval_specs : array-like of shape (n_interval_specs, 3)
            Interval boundaries and split locations ``[start, split, end)``.

        Returns
        -------
        scores : ndarray of shape (n_interval_specs,) or (n_interval_specs, n_features)
            Change scores for each interval specification.
        """
        check_is_fitted(self, ["cost_"])
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            check_sorted=True,
            caller_name=self.__class__.__name__,
        )

        left_intervals = interval_specs[:, [0, 1]]
        right_intervals = interval_specs[:, [1, 2]]
        full_intervals = interval_specs[:, [0, 2]]

        left_costs = self.cost_.evaluate(cache, left_intervals)
        right_costs = self.cost_.evaluate(cache, right_intervals)
        no_change_costs = self.cost_.evaluate(cache, full_intervals)

        change_scores = no_change_costs - (left_costs + right_costs)
        min_score = np.min(change_scores)
        if min_score < -1e-6:
            warnings.warn(
                f"{self.cost.__class__.__name__} produced negative change scores "
                f"(min={min_score:.3g}). The cost may not be subadditive.",
                RuntimeWarning,
                stacklevel=2,
            )
        return np.maximum(change_scores, 0.0)

    @property
    def min_size(self) -> int:
        """Minimum valid interval size inherited from wrapped cost scorer."""
        check_is_fitted(self)
        return self.cost_.min_size

    def get_default_penalty(self) -> float:
        """Get default penalty delegated to the wrapped cost scorer."""
        return self.cost_.get_default_penalty()

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for change score wrapper."""
        tags = super().__sklearn_tags__()
        cost_tags = self.cost.__sklearn_tags__()
        tags.input_tags = cost_tags.input_tags
        tags.interval_scorer_tags.score_type = "change_score"
        tags.interval_scorer_tags.conditional = (
            cost_tags.interval_scorer_tags.conditional
        )
        tags.interval_scorer_tags.aggregated = cost_tags.interval_scorer_tags.aggregated
        tags.interval_scorer_tags.penalised = cost_tags.interval_scorer_tags.penalised
        return tags


# class CostTransientScore(BaseTransientScore):
#     pass
