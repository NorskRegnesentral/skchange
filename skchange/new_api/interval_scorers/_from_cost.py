"""Classes to construct other types of interval scorers from costs."""

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from skchange.new_api.interval_scorers._base import (
    BaseChangeScore,
    BaseIntervalScorer,
)
from skchange.new_api.typing import ArrayLike, Self
from skchange.new_api.utils import SkchangeTags


def to_change_score(
    scorer: BaseIntervalScorer,
    *,
    caller_name: str | None = None,
    arg_name: str = "scorer",
) -> BaseIntervalScorer:
    """Convert a compatible scorer to a change score.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        Scorer to convert.
    caller_name : str or None, default=None
        Caller name used in error messages.
    arg_name : str, default="scorer"
        Argument name used in error messages.

    Returns
    -------
    BaseIntervalScorer
        Scorer with ``score_type='change_score'``.
    """
    score_type = scorer.__sklearn_tags__().interval_scorer_tags.score_type

    if score_type == "change_score":
        return scorer

    if score_type == "cost":
        return CostChangeScore(scorer)

    if caller_name is None:
        caller_name = "to_change_score"
    raise ValueError(
        f"`{arg_name}` in {caller_name} must have score_type 'cost' or "
        f"'change_score'. Got score_type '{score_type}'."
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
        self.cost_: BaseIntervalScorer = clone(self.cost).fit(X, y)
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute wrapped cost scorer data."""
        check_is_fitted(self, ["cost_"])
        return {"cost_cache": self.cost_.precompute(X)}

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
        interval_specs = check_array(interval_specs, ensure_2d=True, dtype=np.int64)
        if interval_specs.shape[1] != 3:
            raise ValueError(
                "interval_specs must have shape (n_interval_specs, 3), "
                f"got {interval_specs.shape}."
            )

        starts = interval_specs[:, 0]
        splits = interval_specs[:, 1]
        ends = interval_specs[:, 2]
        if np.any(splits <= starts) or np.any(splits >= ends):
            raise ValueError(
                "Each interval specification must satisfy start < split < end "
                "for change-score evaluation."
            )

        cost_cache = cache["cost_cache"]
        left_intervals = interval_specs[:, [0, 1]]
        right_intervals = interval_specs[:, [1, 2]]
        full_intervals = interval_specs[:, [0, 2]]

        left_costs = self.cost_.evaluate(cost_cache, left_intervals)
        right_costs = self.cost_.evaluate(cost_cache, right_intervals)
        no_change_costs = self.cost_.evaluate(cost_cache, full_intervals)

        change_scores = no_change_costs - (left_costs + right_costs)
        change_scores[(change_scores < 0) & (change_scores > -1e-8)] = 0.0
        return change_scores

    @property
    def min_size(self) -> int:
        """Minimum valid interval size inherited from wrapped cost scorer."""
        check_is_fitted(self)
        return self.cost_.min_size

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for change score wrapper."""
        tags = super().__sklearn_tags__()
        cost_tags = self.cost.__sklearn_tags__().interval_scorer_tags
        tags.interval_scorer_tags.score_type = "change_score"
        tags.interval_scorer_tags.conditional = cost_tags.conditional
        tags.interval_scorer_tags.aggregated = cost_tags.aggregated
        tags.interval_scorer_tags.penalised = cost_tags.penalised
        return tags


# class CostTransientScore(BaseTransientScore):
#     pass
