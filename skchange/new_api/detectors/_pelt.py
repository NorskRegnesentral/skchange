"""The pruned exact linear time (PELT) algorithm."""

__author__ = ["Tveten", "johannvk"]
__all__ = ["PELT"]

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.detectors._base import BaseChangeDetector
from skchange.new_api.interval_scorers._base import BaseCost
from skchange.new_api.interval_scorers._costs.l2_cost import L2Cost
from skchange.new_api.typing import ArrayLike, Self
from skchange.new_api.utils import SkchangeTags
from skchange.new_api.utils._param_validation import HasMethods, Interval, _fit_context
from skchange.new_api.utils.validation import check_interval_scorer, validate_data
from skchange.utils.numba import njit


@dataclass(frozen=True, kw_only=True, eq=False)
class PELTResult:
    """Result of running the PELT algorithm.

    Containing:
    - `optimal_costs`: The optimal penalized segmentation costs for each sample.
    - `previous_change_points`: The optimal previous change point for each sample.
    - `pruning_fraction`: The fraction of starts pruned during the run, as compared
                          to Optimal Partitioning.
    - `changepoints`: The final set of changepoints.
    """

    optimal_costs: np.ndarray
    previous_change_points: np.ndarray
    pruning_fraction: float
    changepoints: np.ndarray

    def __eq__(self, other):
        """Check equality between two PELTResult instances.

        Compares all attributes using numpy's array_equal for array attributes.
        """
        if not isinstance(other, PELTResult):
            return False

        return (
            np.array_equal(self.optimal_costs, other.optimal_costs)
            and np.array_equal(
                self.previous_change_points, other.previous_change_points
            )
            and self.pruning_fraction == other.pruning_fraction
            and np.array_equal(self.changepoints, other.changepoints)
        )

    @classmethod
    def new(
        cls,
        optimal_costs: np.ndarray,
        previous_change_points: np.ndarray,
        pruning_fraction: float,
    ) -> "PELTResult":
        """Create a new PeltResult instance."""
        # Check that the lengths of opt_cost and prev_cpts match:
        if len(optimal_costs) != len(previous_change_points):
            raise ValueError(
                "All input arrays must have the same length. "
                "The lengths of `opt_cost` and `prev_cpts` were "
                f"{len(optimal_costs)} != {len(previous_change_points)}."
            )
        changepoints = get_changepoints(previous_change_points)
        return cls(
            optimal_costs=optimal_costs,
            previous_change_points=previous_change_points,
            pruning_fraction=pruning_fraction,
            changepoints=changepoints,
        )


@njit
def get_changepoints(prev_cpts: np.ndarray) -> np.ndarray:
    changepoints = []
    i = len(prev_cpts) - 1
    while i >= 0:
        cpt_i = prev_cpts[i]
        changepoints.append(cpt_i)
        i = cpt_i - 1
    return np.array(changepoints[-2::-1])  # Remove the artificial changepoint at 0.


def _run_pelt(
    cost: BaseCost,
    X: np.ndarray,
    penalty: float,
    min_segment_length: int,
    split_cost: float = 0.0,
    prune: bool = True,
    pruning_margin: float = 0.0,
) -> PELTResult:
    """Run the PELT algorithm.

    Currently agrees with the 'changepoint::cpt.mean' implementation of PELT in R.
    If the 'min_segment_length' is large enough to span more than a single changepoint,
    the algorithm can return a suboptimal partitioning.
    In that case, resort to the 'optimal_partitioning' algorithm.

    Contract:
    - The `cost` will never be evaluated on intervals shorter than `min_segment_length`.

    Parameters
    ----------
    cost: BaseCost
        The cost to use.
    penalty : float
        The penalty incurred for adding a changepoint.
    min_segment_length : int
        The minimum length of a segment, by default 1.
    split_cost : float, optional
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the above inequality.
    prune: bool, optional
        If False, drop the pruning step, reverting to optimal partitioning.
        Can be useful for debugging and testing. By default set to True.
    pruning_margin : float, optional
        The pruning margin to use. By default set to zero.
        This is used to reduce pruning of the admissible starts set.
        Can be useful if the cost function is imprecise, i.e.
        based on solving an optimization problem with a large tolerance.

    Returns
    -------
    PELTResult
        Container for the results of the PELT algorithm run.
    """
    check_is_fitted(cost)
    cache = cost.precompute(X)
    n_samples = cost.n_samples_in_

    if min_segment_length > n_samples:
        raise ValueError(
            "The `min_segment_length` cannot be larger than the number of samples."
        )

    # Store the previous changepoint for each latest start added.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = np.repeat(0, n_samples)

    min_segment_shift = min_segment_length - 1

    opt_cost = np.concatenate((np.array([-penalty]), np.zeros(n_samples)))

    # Cannot compute the cost for the first 'min_segment_shift' elements:
    opt_cost[1 : min(min_segment_length, n_samples)] = np.inf

    num_pelt_cost_evals = 0
    num_opt_part_cost_evals = 0

    # Compute the optimal cost for indices
    # [min_segment_length, 2*min_segment_length - 1] directly:
    non_changepoint_slice_end = min(2 * min_segment_length, n_samples + 1)
    non_changepoint_ends = np.arange(min_segment_length, non_changepoint_slice_end)
    non_changepoint_starts = np.zeros(len(non_changepoint_ends), dtype=np.int64)
    non_changepoint_intervals = np.column_stack(
        (non_changepoint_starts, non_changepoint_ends)
    )

    # TODO: Only allow aggregated costs in to "PELT"? User decides aggregation method.
    non_changepoint_costs = np.sum(
        cost.evaluate(cache, non_changepoint_intervals), axis=1
    )
    opt_cost[min_segment_length:non_changepoint_slice_end] = non_changepoint_costs

    num_pelt_cost_evals += len(non_changepoint_starts)
    num_opt_part_cost_evals += len(non_changepoint_starts)

    # Evolving set of admissible segment starts.
    cost_eval_starts = np.array(([0]), dtype=np.int64)

    potential_change_point_indices = np.arange(2 * min_segment_length - 1, n_samples)

    # Add a buffer for pruning indices: Start as empty arrays.
    pruning_indices = [np.array([]) for _ in range(min_segment_length)]

    # Triangle number forumla for the unpruned number of cost evaluations:
    num_opt_part_cost_evals += (len(potential_change_point_indices) + 2) * (
        len(potential_change_point_indices) + 1
    ) // 2 - 1

    for current_obs_ind in potential_change_point_indices:
        latest_start = current_obs_ind - min_segment_shift
        opt_cost_obs_ind = current_obs_ind + 1

        if prune:
            starts_to_prune = pruning_indices[current_obs_ind % min_segment_length]
            # Delete the start indices that can be pruned:
            cost_eval_starts = np.delete(
                cost_eval_starts,
                np.where(np.isin(cost_eval_starts, starts_to_prune))[0],
            )

        # Add the next start to the admissible starts set:
        cost_eval_starts = np.concatenate((cost_eval_starts, np.array([latest_start])))
        cost_eval_ends = np.repeat(current_obs_ind + 1, len(cost_eval_starts))
        cost_eval_intervals = np.column_stack((cost_eval_starts, cost_eval_ends))
        interval_costs = np.sum(cost.evaluate(cache, cost_eval_intervals), axis=1)

        num_pelt_cost_evals += len(cost_eval_starts)

        # Add the cost and penalty for a new segment:
        candidate_opt_costs = opt_cost[cost_eval_starts] + interval_costs + penalty

        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        opt_cost[opt_cost_obs_ind] = candidate_opt_costs[argmin_candidate_cost]
        prev_cpts[current_obs_ind] = cost_eval_starts[argmin_candidate_cost]

        if prune:
            # Trimming the admissible starts set: (reuse the array of optimal costs)
            current_obs_ind_opt_cost = opt_cost[opt_cost_obs_ind]

            abs_current_obs_opt_cost = np.abs(current_obs_ind_opt_cost)
            start_inclusion_threshold = (
                current_obs_ind_opt_cost
                # Apply pruning margin to the current optimal cost:
                + abs_current_obs_opt_cost * pruning_margin
                # Moved from 'negative' on left side
                # to 'positive' on right side.
                + penalty
                # Remove from right side of inequality.
                - split_cost
            )

            # Store indices to prune for the `min_segment_length`'th next observation:
            pruning_indices[current_obs_ind % min_segment_length] = cost_eval_starts[
                candidate_opt_costs > start_inclusion_threshold
            ]

    pruning_fraction = (
        1.0 - num_pelt_cost_evals / num_opt_part_cost_evals
        if num_opt_part_cost_evals > 0
        else np.nan
    )

    pelt_result = PELTResult.new(
        optimal_costs=opt_cost[1:],
        previous_change_points=prev_cpts,
        pruning_fraction=pruning_fraction,
    )

    return pelt_result


def _run_pelt_min_segment_length_one(
    cost: BaseCost,
    X: np.ndarray,
    penalty: float,
    split_cost: float = 0.0,
    prune: bool = True,
    pruning_margin: float = 0.0,
) -> PELTResult:
    """Run the PELT algorithm, with a minimum segment length of one.

    This is a special case of the PELT algorithm, where the minimum segment length
    is set to one. This means that the algorithm can consider every single point as a
    potential changepoint, and thus it can be used for very fine-grained change point
    detection. We can also simplify the algorithm by not having to keep track of
    deferred pruning information, as all pruning of start points is applicable
    for the next observation.

    Parameters
    ----------
    cost: BaseCost
        The cost to use.
    X : np.ndarray of shape (n_samples, n_features)
        Input data. ``cost`` must already be fitted to ``X``.
    penalty : float
        The penalty incurred for adding a changepoint.
    split_cost : float, optional
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the
        above inequality.
    prune: bool, optional
        If False, drop the pruning step, performing optimal partitioning.
        Can be useful for debugging and testing. By default set to True.
    pruning_margin : float, optional
        The pruning margin to use. By default set to zero.
        This is used to reduce pruning of the admissible starts set.
        Can be useful if the cost function is imprecise, i.e.
        based on solving an optimization problem with large tolerance.

    Returns
    -------
    PELTResult
        Summary of the PELT algorithm run, containing:
        - `optimal_cost`: The optimal costs for each segment.
        - `previous_change_points`: The previous changepoints for each segment.
        - `pruning_fraction`: The fraction of pruning applied during the run.
        - `changepoints`: The final set of changepoints.
    """
    check_is_fitted(cost)
    cache = cost.precompute(X)
    n_samples = cost.n_samples_in_
    if n_samples < 1:
        raise ValueError(
            "The number of samples for the fitted cost must be at least one. "
            f"Got {n_samples} samples."
        )

    opt_cost = np.concatenate((np.array([-penalty]), np.zeros(n_samples)))

    # Compute the cost for the first observation directly:
    opt_cost[1] = cost.evaluate(cache, np.array([[0, 1]]))[0, 0]

    # Aggregate number of cost evaluations:
    num_pelt_cost_evals = 1
    num_opt_part_cost_evals = 1

    # Store the previous changepoint for each latest start added.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = np.repeat(0, n_samples)

    # Evolving set of admissible segment starts.
    eval_starts = np.array(([0]), dtype=np.int64)

    observation_indices = np.arange(1, n_samples)

    num_opt_part_cost_evals += (len(observation_indices) + 2) * (
        len(observation_indices) + 1
    ) // 2 - 1

    for current_obs_ind in observation_indices:
        opt_cost_obs_ind = current_obs_ind + 1

        # Add the next start to the admissible starts set:
        eval_starts = np.concatenate((eval_starts, np.array([current_obs_ind])))
        eval_ends = np.repeat(current_obs_ind + 1, len(eval_starts))
        eval_intervals = np.column_stack((eval_starts, eval_ends))
        interval_costs = np.sum(cost.evaluate(cache, eval_intervals), axis=1)

        num_pelt_cost_evals += len(eval_starts)

        # Add the cost and penalty for a new segment:
        candidate_opt_costs = opt_cost[eval_starts] + interval_costs + penalty

        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        opt_cost[opt_cost_obs_ind] = candidate_opt_costs[argmin_candidate_cost]
        prev_cpts[current_obs_ind] = eval_starts[argmin_candidate_cost]

        if prune:
            # Trimming the admissible starts set: (reuse the array of optimal costs)
            current_obs_ind_opt_cost = opt_cost[opt_cost_obs_ind]

            abs_current_obs_opt_cost = np.abs(current_obs_ind_opt_cost)
            start_inclusion_threshold = (
                current_obs_ind_opt_cost
                # Apply pruning margin to the current optimal cost:
                + abs_current_obs_opt_cost * pruning_margin
                # Moved from 'negative' on left side
                # to 'positive' on right side.
                + penalty
                # Remove from right side of inequality.
                - split_cost
            )

            # Apply pruning:
            eval_starts = eval_starts[candidate_opt_costs <= start_inclusion_threshold]

    pruning_fraction = (
        1.0 - num_pelt_cost_evals / num_opt_part_cost_evals
        if num_opt_part_cost_evals > 0
        else np.nan
    )

    pelt_result = PELTResult.new(
        optimal_costs=opt_cost[1:],
        previous_change_points=prev_cpts,
        pruning_fraction=pruning_fraction,
    )

    return pelt_result


def _run_pelt_with_step_size(
    cost: BaseCost,
    X: np.ndarray,
    penalty: float,
    step_size: int,
    split_cost: float = 0.0,
    prune: bool = True,
    pruning_margin: float = 0.0,
) -> PELTResult:
    """Run the PELT algorithm.

    Solves the PELT optimization problem where only indices that are multiples of
    `step_size` from the start (index `0`) are considered as potential changepoints.
    This means that the minimum segment length is naturally `step_size`.

    Parameters
    ----------
    cost: BaseCost
        The cost to use.
    X : np.ndarray of shape (n_samples, n_features)
        Input data. ``cost`` must already be fitted to ``X``.
    penalty : float
        The penalty incurred for adding a changepoint.
    step_size : int
        Only indices that are multiples of `step_size` from the start (index `0`) are
        considered as potential changepoints. This also means that the minimum segment
        length is naturally `step_size`.
    split_cost : float, optional
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the
        above inequality.
    prune: bool, optional
        If False, drop the pruning step, reverting to optimal partitioning.
        Can be useful for debugging and testing. By default set to True.
    pruning_margin : float, optional
        The pruning margin to use. By default set to zero.
        This is used to reduce pruning of the admissible starts set.
        Can be useful if the cost function is imprecise, i.e.
        based on solving an optimization problem with large tolerance.

    Returns
    -------
    PELTResult
        Container for the results of the PELT algorithm run.
    """
    check_is_fitted(cost)
    cache = cost.precompute(X)
    n_samples = cost.n_samples_in_
    if n_samples < step_size:
        raise ValueError("The `step_size` cannot be larger than the number of samples.")

    # Initialize the optimal costs array:
    opt_cost = np.concatenate((np.array([-penalty]), np.zeros(n_samples)))

    # Store the previous changepoint for each latest start added.
    # Used to get the final set of changepoints after the loop.
    # Initialize to invalid previous changepoints:
    prev_cpts = np.zeros(n_samples, dtype=np.int64)

    # Evolving set of admissible segment starts.
    eval_starts = np.array([], dtype=np.int64)

    observation_interval_starts = np.arange(
        start=0, stop=n_samples - step_size + 1, step=step_size
    )
    observation_interval_ends = np.concatenate(
        (
            np.arange(start=step_size - 1, stop=n_samples - step_size, step=step_size),
            np.array([n_samples - 1]),
        )
    )
    observation_intervals = np.column_stack(
        (observation_interval_starts, observation_interval_ends)
    )

    # Triangle number formula for the unpruned number of cost evaluations.
    opt_part_cost_evals = (
        len(observation_intervals) * (len(observation_intervals) + 1) // 2
    )
    pelt_cost_evals = 0

    for obs_interval_start, obs_interval_end in observation_intervals:
        # Add the next start to the admissible starts set:
        eval_starts = np.concatenate((eval_starts, np.array([obs_interval_start])))
        eval_ends = np.repeat(obs_interval_end + 1, len(eval_starts))
        eval_intervals = np.column_stack((eval_starts, eval_ends))
        interval_costs = np.sum(cost.evaluate(cache, eval_intervals), axis=1)

        pelt_cost_evals += len(eval_starts)

        # Add the penalty for a new segment:
        candidate_opt_costs = opt_cost[eval_starts] + interval_costs + penalty

        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        opt_cost[obs_interval_start + 1 : obs_interval_end + 1 + 1] = (
            candidate_opt_costs[argmin_candidate_cost]
        )
        prev_cpts[obs_interval_start : obs_interval_end + 1] = eval_starts[
            argmin_candidate_cost
        ]

        if prune:
            # Trimming the admissible starts set: (reuse the array of optimal costs)
            current_obs_ind_opt_cost = opt_cost[obs_interval_start + 1]

            abs_current_obs_opt_cost = np.abs(current_obs_ind_opt_cost)
            start_inclusion_threshold = (
                current_obs_ind_opt_cost
                # Apply pruning margin to the current optimal cost:
                + abs_current_obs_opt_cost * pruning_margin
                # Moved from 'negative' on left side
                # to 'positive' on right side.
                + penalty
                # Remove from right side of inequality.
                - split_cost
            )

            new_start_inclusion_mask = candidate_opt_costs <= start_inclusion_threshold
            eval_starts = eval_starts[new_start_inclusion_mask]

    pruning_fraction = (
        (1.0 - pelt_cost_evals / opt_part_cost_evals)
        if opt_part_cost_evals > 0
        else np.nan
    )

    pelt_result = PELTResult.new(
        optimal_costs=opt_cost[1:],
        previous_change_points=prev_cpts,
        pruning_fraction=pruning_fraction,
    )

    return pelt_result


def _resolve_cost(cost: BaseCost | None) -> BaseCost:
    """Return cost or the default L2Cost().

    Needed since default resolution needs to be done in both fit and __sklearn_tags__
    to ensure correct input tags are propagated.
    """
    return cost if cost is not None else L2Cost()


class PELT(BaseChangeDetector):
    """Pruned exact linear time (PELT) changepoint detection.

    Implements the PELT algorithm [1]_ for changepoint detection.
    This method solves the penalized optimal partitioning problem,
    with pruning of the admissible starts set applied to improve performance.

    One can specify a minimum segment length for the partitions considered
    when detecting changepoints through the `min_segment_length` parameter,
    and when the minimum segment length is greater than one we use deferred
    pruning of the admissible starts [2]_ to ensure exact solutions.

    Additionally, one can specify a step size through the `step_size` parameter,
    which coarsens the search space for changepoints, allowing for faster detection
    at the cost of change point location granularity.

    Parameters
    ----------
    cost : BaseCost or None, default=None
        Cost to use for the changepoint detection. Must be a ``BaseCost``
        instance with ``score_type='cost'``. Passing a ``PenalisedScore``
        will raise a ``ValueError`` in ``fit``.
        If ``None``, defaults to ``L2Cost()``.
    penalty : float or None, default=None
        Penalty incurred for each added changepoint. Must be non-negative.
        If ``None``, defaults to ``cost_.get_default_penalty()`` after fitting
        (BIC-based penalty).
    min_segment_length : int, default=1
        Minimum number of samples in a segment. Must be at least 1. If
        ``step_size > 1``, this must be less than or equal to ``step_size``.
    step_size : int, default=1
        Only indices that are multiples of ``step_size`` from the start are
        considered as potential changepoints. Implicitly ensures that
        ``min_segment_length >= step_size``, but it is an error to specify
        ``min_segment_length`` greater than ``step_size``.
    split_cost : float, default=0.0
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the above inequality.
    prune : bool, default=True
        If False, drop the pruning step. Reverts to optimal partitioning.
        Can be useful for debugging and testing.
    pruning_margin : float, default=0.0
        The pruning margin to use. By default set to zero.
        This is used to reduce pruning of the admissible starts set.
        Can be useful if the cost function is imprecise, i.e.
        based on solving an optimization problem with large tolerance.

    Attributes
    ----------
    cost_ : BaseCost
        Fitted cost scorer.
    penalty_ : float
        Penalty value used (either user-specified or default from ``cost_``).

    References
    ----------
    .. [1] Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of
       changepoints with a linear computational cost. Journal of the American
       Statistical Association, 107(500), 1590-1598.

    .. [2] Bakka, Kristin Benedicte (2018). Changepoint model selection in Gaussian
       data by maximization of approximate Bayes Factors with the Pruned Exact Linear
       Time algorithm. Master's thesis, Norwegian University of Science and Technology
       (NTNU). URL: https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2558597.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.detectors import PELT
    >>> rng = np.random.default_rng(2)
    >>> X = np.concatenate([rng.normal(0, 1, (100, 1)),
    ...                     rng.normal(10, 1, (100, 1))])
    >>> detector = PELT()
    >>> detector.fit(X).predict_changepoints(X)
    array([100])
    """

    _parameter_constraints = {
        "cost": [HasMethods(["fit", "precompute", "evaluate"]), None],
        "penalty": [Interval(Real, 0, None, closed="left"), None],
        "min_segment_length": [Interval(Integral, 1, None, closed="left")],
        "step_size": [Interval(Integral, 1, None, closed="left")],
        "split_cost": [Interval(Real, 0, None, closed="left")],
        "prune": ["boolean"],
        "pruning_margin": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        cost: BaseCost | None = None,
        penalty: float | None = None,
        min_segment_length: int = 1,
        step_size: int = 1,
        split_cost: float = 0.0,
        prune: bool = True,
        pruning_margin: float = 0.0,
    ):
        self.cost = cost
        self.penalty = penalty
        self.min_segment_length = min_segment_length
        self.step_size = step_size
        self.split_cost = split_cost
        self.prune = prune
        self.pruning_margin = pruning_margin

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get tags, propagating input constraints from the cost."""
        tags = super().__sklearn_tags__()
        tags.input_tags = _resolve_cost(self.cost).__sklearn_tags__().input_tags
        return tags

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the cost to training data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training time series data.
        y : ArrayLike | None, default=None
            Ignored.

        Returns
        -------
        self : PELT
            Fitted detector.
        """
        X = validate_data(self, X, reset=True, ensure_2d=True)

        if self.step_size > 1 and self.min_segment_length > self.step_size:
            raise ValueError(
                f"`min_segment_length` (={self.min_segment_length}) cannot be "
                f"greater than `step_size` (={self.step_size}) when step_size > 1."
            )

        cost = _resolve_cost(self.cost)
        check_interval_scorer(
            cost,
            ensure_score_type=["cost"],
            allow_penalised=False,
            caller_name=self.__class__.__name__,
            arg_name="cost",
        )
        self.cost_ = clone(cost).fit(X, y)
        self.penalty_ = (
            self.cost_.get_default_penalty() if self.penalty is None else self.penalty
        )

        return self

    def predict_changepoints(self, X: ArrayLike) -> np.ndarray:
        """Detect changepoints in a time series.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyse for changepoints.

        Returns
        -------
        changepoints : np.ndarray of shape (n_changepoints,)
            Sorted integer indices of detected changepoints.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True)

        cost = clone(self.cost_).fit(X)

        if self.step_size > 1:
            pelt_result = _run_pelt_with_step_size(
                cost=cost,
                X=X,
                penalty=self.penalty_,
                step_size=self.step_size,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
            )
        elif self.min_segment_length == 1:
            pelt_result = _run_pelt_min_segment_length_one(
                cost=cost,
                X=X,
                penalty=self.penalty_,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
            )
        else:
            pelt_result = _run_pelt(
                cost=cost,
                X=X,
                penalty=self.penalty_,
                min_segment_length=self.min_segment_length,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
            )

        return pelt_result.changepoints.astype(np.intp)
