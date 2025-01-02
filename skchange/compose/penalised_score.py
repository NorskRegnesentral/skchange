"""Penalised interval scorer."""

import numpy as np

from skchange.base import BaseIntervalScorer
from skchange.penalties import BasePenalty
from skchange.utils.numba import njit


@njit
def penalise_scores_constant(
    scores: np.ndarray, penalty_values: np.ndarray
) -> np.ndarray:
    """Penalise scores with a constant penalty.

    Parameters
    ----------
    scores : np.ndarray
        The scores to penalise. The output of a BaseIntervalScorer.
    penalty_values : np.ndarray
        The penalty values. The output of a constant BasePenalty.

    Returns
    -------
    penalised_scores : np.ndarray
        The penalised scores.
    """
    if penalty_values.size != 1:
        raise ValueError(
            "The penalty must have a single value for the constant penalty."
        )

    penalised_scores = scores.sum(axis=1) - penalty_values
    return penalised_scores


@njit
def _check_nonconstant_penalties(
    penalty_values: np.ndarray, scores: np.ndarray
) -> None:
    if penalty_values.size != scores.shape[1]:
        raise ValueError("The penalty must have a value for each variable in the data.")
    if penalty_values.size < 2:
        raise ValueError(
            "The penalty must have at least 2 values for the linear penalty."
        )


@njit
def penalise_scores_linear(
    scores: np.ndarray, penalty_values: np.ndarray
) -> np.ndarray:
    """Penalise scores with a linear penalty.

    Parameters
    ----------
    scores : np.ndarray
        The scores to penalise. The output of a BaseIntervalScorer.
    penalty_values : np.ndarray
        The penalty values. The output of a linear BasePenalty.

    Returns
    -------
    penalised_savings : np.ndarray
        The penalised savings
    """
    _check_nonconstant_penalties(penalty_values, scores)

    penalty_slope = penalty_values[1] - penalty_values[0]
    penalty_intercept = penalty_values[0] - penalty_slope

    penalised_scores_matrix = (
        np.maximum(scores - penalty_slope, 0.0) - penalty_intercept
    )
    penalised_savings = penalised_scores_matrix.sum(axis=1)
    return penalised_savings


@njit
def penalise_scores_nonlinear(
    scores: np.ndarray, penalty_values: np.ndarray
) -> np.ndarray:
    """Penalise scores with a nonlinear penalty.

    Parameters
    ----------
    scores : np.ndarray
        The scores to penalise. The output of a BaseIntervalScorer.
    penalty_values : np.ndarray
        The penalty values. The output of a nonlinear BasePenalty.

    Returns
    -------
    penalised_scores : np.ndarray
        The penalised scores
    """
    _check_nonconstant_penalties(penalty_values, scores)

    penalised_scores = []
    for score in scores:
        sorted_scores = np.sort(score)[::-1]
        penalised_score = np.cumsum(sorted_scores) - penalty_values
        optimal_penalised_score = np.max(penalised_score)
        penalised_scores.append(optimal_penalised_score)
    return np.array(penalised_scores, dtype=np.float64)


class PenalisedScore(BaseIntervalScorer):
    """Penalised interval scorer.

    Penalises the scores of an interval scorer and aggregates them into a single value
    for each cut. For non-constant penalties, the penalised score is optimised over the
    number of affected components.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        The interval scorer to penalise.
    penalty : BasePenalty
        The penalty to apply to the scores. The penalty must be constant for
        multivariate scorers. If the penalty is already fitted, it will not be refitted
        to the data in the `fit` method.
    """

    evaluation_type = "multivariate"

    def __init__(self, scorer: BaseIntervalScorer, penalty: BasePenalty):
        self.scorer = scorer
        self.penalty = penalty
        super().__init__()

        self.expected_cut_entries = scorer.expected_cut_entries

        if (
            scorer.evaluation_type == "multivariate"
            and penalty.penalty_type != "constant"
        ):
            raise ValueError(
                "Multivariate scorers are only compatible with constant penalties."
            )

    def _fit(self, X: np.ndarray, y=None) -> "PenalisedScore":
        """Fit the penalised interval scorer to training data.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y : None
            Ignored. Included for API consistency by convention.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        If the penalty is already fitted, it will not be refitted to the data. If the
        data is not compatible with the penalty, a ValueError will be raised.
        """
        self.scorer.fit(X)

        if self.penalty.is_fitted:
            if X.shape[1] != self.penalty.p:
                raise ValueError(
                    "The number of variables in the data must match the number of"
                    " variables in the penalty."
                    f" 'X.shape[1]' = {X.shape[1]} and 'penalty.p' = {self.penalty.p}."
                    " This error is most likely due to the penalty being fitted to a "
                    " different data set than the scorer."
                )
        else:
            self.penalty.fit(X, self.scorer)

        if self.penalty.penalty_type == "constant" or X.shape[1] == 1:
            self.penalise_scores = penalise_scores_constant
        elif self.penalty.penalty_type == "linear":
            self.penalise_scores = penalise_scores_linear
        elif self.penalty.penalty_type == "nonlinear":
            self.penalise_scores = penalise_scores_nonlinear
        else:
            raise ValueError(
                f"Unknown penalty type '{self.penalty.penalty_type}'."
                " Must be 'constant', 'linear' or 'nonlinear'."
            )

        return self

    def _evaluate(self, cuts: np.ndarray) -> np.ndarray:
        """Evaluate the penalised scores according to a set of cuts.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array of integer location-based cuts to evaluate. Each row in the array
            must be sorted in increasing order.

        Returns
        -------
        values : np.ndarray
            A 2D array of scores. One row for each row in cuts.
        """
        scores = self.scorer.evaluate(cuts)
        return self.penalise_scores(scores, self.penalty.values).reshape(-1, 1)

    @property
    def min_size(self) -> int:
        """Minimum valid size of an interval to evaluate.

        The size of each interval is by default defined as ``np.diff(cuts[i, ])``.
        Subclasses can override the min_size to mean something else, for example in
        cases where intervals are combined before evaluation or `cuts` specify
        disjoint intervals.

        Returns
        -------
        int or None
            The minimum valid size of an interval to evaluate. If ``None``, it is
            unknown what the minimum size is. E.g., the scorer may need to be fitted
            first to determine the minimum size.
        """
        return self.scorer.min_size

    def get_param_size(self, p: int) -> int:
        """Get the number of parameters to estimate over each interval.

        The primary use of this method is to determine an appropriate default penalty
        value in detectors. For example, a scorer for a change in mean has one
        parameter to estimate per variable in the data, a scorer for a change in the
        mean and variance has two parameters to estimate per variable, and so on.
        Subclasses should override this method accordingly.

        Parameters
        ----------
        p : int
            Number of variables in the data.
        """
        return self.scorer.get_param_size(p)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for interval scorers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from skchange.anomaly_scores import L2Saving
        from skchange.change_scores import MultivariateGaussianScore
        from skchange.penalties import BICPenalty, LinearChiSquarePenalty

        params = [
            {"scorer": L2Saving(), "penalty": LinearChiSquarePenalty()},
            {"scorer": MultivariateGaussianScore(), "penalty": BICPenalty()},
        ]

        return params
