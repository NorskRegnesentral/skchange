import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseIntervalScorer
from skchange.new_api.typing import ArrayLike, Self
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import (
    check_interval_scorer,
    check_penalty,
    validate_data,
)
from skchange.utils.numba import njit


@njit
def _penalise_scores_constant(scores: np.ndarray, penalty: float) -> np.ndarray:
    """Penalise scores with a constant penalty."""
    return scores.sum(axis=1) - penalty


@njit
def _penalise_scores_linear(
    scores: np.ndarray, penalty_values: np.ndarray
) -> np.ndarray:
    """Penalise scores with linear penalty values."""
    penalty_slope = penalty_values[1] - penalty_values[0]
    penalty_intercept = penalty_values[0] - penalty_slope
    penalised_scores_matrix = (
        np.maximum(scores - penalty_slope, 0.0) - penalty_intercept
    )
    return penalised_scores_matrix.sum(axis=1)


@njit
def _penalise_scores_nonlinear(
    scores: np.ndarray, penalty_values: np.ndarray
) -> np.ndarray:
    """Penalise scores with nonlinear penalty values."""
    penalised_scores = []
    for score in scores:
        sorted_scores = np.sort(score)[::-1]
        penalised_score = np.cumsum(sorted_scores) - penalty_values
        optimal_penalised_score = np.max(penalised_score)
        penalised_scores.append(optimal_penalised_score)
    return np.array(penalised_scores, dtype=np.float64)


class PenalisedScore(BaseIntervalScorer):
    """Penalised interval scorer wrapper for new API scorers.

    Aggregates feature-wise scores and applies either constant, linear, or
    nonlinear penalties over the number of affected variables.

    Let sorted_score be the k-th largest feature-wise score for an interval, then
    the penalised score is computed as
    ``np.max(np.cumsum(sorted_scores) - penalty_values)``.

    For a constant penalty and a linear penalty, this reduces to simpler forms that can
    be computed more efficiently.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        The base interval scorer to wrap and penalise.
    penalty : array-like of shape (n_features,), float, or None, default=None
        Penalty values. `penalty[k]` is the penalty for including k features in the
        aggregated penalised score. If float, the value is broadcast across all k.

    """

    def __init__(
        self,
        scorer: BaseIntervalScorer,
        penalty: ArrayLike | float | None = None,
    ):
        self.scorer = scorer
        self.penalty = penalty

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit wrapped scorer and select penalty mode."""
        X = validate_data(
            self,
            X,
            ensure_2d=True,
            dtype=np.float64,
            reset=True,
        )

        check_interval_scorer(
            self.scorer,
            ensure_score_type=["change_score", "saving", "transient_score"],
            allow_penalised=False,
            caller_name=self.__class__.__name__,
            arg_name="scorer",
        )
        self.scorer_ = clone(self.scorer)
        self.scorer_.fit(X, y)

        scorer_tags = self.scorer_.__sklearn_tags__().interval_scorer_tags
        if scorer_tags.aggregated and self.penalty is not None:
            penalty_arr = np.asarray(self.penalty).reshape(-1)
            if penalty_arr.size > 1:
                raise ValueError(
                    "`penalty` must be scalar for aggregated input scores."
                )

        penalty = (
            self.scorer_.get_default_penalty() if self.penalty is None else self.penalty
        )
        self.penalty_ = check_penalty(
            penalty,
            caller_name=self.__class__.__name__,
            arg_name="penalty",
        )

        penalty_values = np.asarray(self.penalty_).reshape(-1)
        if penalty_values.size > 1 and penalty_values.size != X.shape[1]:
            raise ValueError(
                "`penalty` must be scalar or have length equal to n_features. "
                f"Got penalty length {penalty_values.size} and n_features {X.shape[1]}."
            )

        if X.shape[1] == 1 or penalty_values.size == 1:
            self._penalty_mode = "constant"
        elif np.allclose(np.diff(penalty_values), np.diff(penalty_values)[0]):
            self._penalty_mode = "linear"
        else:
            self._penalty_mode = "nonlinear"

        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute wrapped scorer data for penalised evaluation."""
        check_is_fitted(self, ["scorer_", "penalty_", "_penalty_mode"])
        return self.scorer_.precompute(X)

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate penalised scores on interval specifications.

        Parameters
        ----------
        cache : dict
            Cache from precompute().
        interval_specs : array-like
            Each row specifies an interval and possibly split points, depending on
            the wrapped scorer type. The expected shape is determined by
            ``self.scorer_``.

        Returns
        -------
        scores : ndarray of shape (n_interval_specs, 1)
            Penalised, aggregated score for each interval specification.
        """
        check_is_fitted(self, ["scorer_", "penalty_", "_penalty_mode"])

        scores = self.scorer_.evaluate(
            cache,
            interval_specs,
        )
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)

        if self._penalty_mode == "constant":
            penalty_value = float(np.asarray(self.penalty_).reshape(-1)[0])
            penalised = _penalise_scores_constant(scores, penalty_value)
        elif self._penalty_mode == "linear":
            penalty_values = np.asarray(self.penalty_, dtype=np.float64).reshape(-1)
            penalised = _penalise_scores_linear(scores, penalty_values)
        elif self._penalty_mode == "nonlinear":
            penalty_values = np.asarray(self.penalty_, dtype=np.float64).reshape(-1)
            penalised = _penalise_scores_nonlinear(scores, penalty_values)
        else:
            raise RuntimeError(f"Unknown penalty mode: {self._penalty_mode}")

        return penalised.reshape(-1, 1)

    @property
    def interval_specs_ncols(self) -> int:
        """Expected width of interval specifications inherited from wrapped scorer."""
        return self.scorer_.interval_specs_ncols

    @property
    def min_size(self) -> int:
        """Minimum valid interval size inherited from wrapped scorer."""
        check_is_fitted(self)
        return self.scorer_.min_size

    def get_default_penalty(self) -> float | np.ndarray:
        """Get the default penalty for the fitted scorer.

        Returns
        -------
        float or np.ndarray
            Default penalty value. Scalar for univariate/constant penalties,
            array of shape ``(n_features,)`` for multivariate penalties.
        """
        check_is_fitted(self)
        return self.scorer_.get_default_penalty()

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get sklearn-compatible tags for penalised scorer."""
        tags = super().__sklearn_tags__()
        scorer_tags = self.scorer.__sklearn_tags__()
        tags.input_tags = scorer_tags.input_tags
        tags.interval_scorer_tags.score_type = (
            scorer_tags.interval_scorer_tags.score_type
        )
        tags.interval_scorer_tags.aggregated = True
        tags.interval_scorer_tags.penalised = True
        return tags
