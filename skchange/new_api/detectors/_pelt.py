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
from skchange.new_api.types import ArrayLike, Self
from skchange.new_api.utils import SkchangeTags
from skchange.new_api.utils._numba import njit
from skchange.new_api.utils._param_validation import HasMethods, Interval, _fit_context
from skchange.new_api.utils.validation import (
    check_interval_scorer,
    skip_validation,
    validate_data,
)


@dataclass(frozen=True, kw_only=True, eq=False)
class PELTResult:
    """Result of running the PELT algorithm.

    Containing:
    - `optimal_costs`: The optimal penalized segmentation costs for each sample.
    - `previous_change_points`: The optimal previous change point for each sample.
    - `pruning_fraction`: The fraction of starts pruned during the run, as compared
                          to Optimal Partitioning.
    - `changepoints`: The final set of changepoints.
    - `interval_starts`, `interval_ends`, `interval_costs`: Flat arrays giving every
      ``(start, end)`` interval on which the DP evaluated the cost, with its
      (feature-summed) cost value. Used by ``predict_scores`` for introspection
      and penalty calibration.
    """

    optimal_costs: np.ndarray
    previous_change_points: np.ndarray
    pruning_fraction: float
    changepoints: np.ndarray
    interval_starts: np.ndarray
    interval_ends: np.ndarray
    interval_costs: np.ndarray

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
            and np.array_equal(self.interval_starts, other.interval_starts)
            and np.array_equal(self.interval_ends, other.interval_ends)
            and np.array_equal(self.interval_costs, other.interval_costs)
        )

    @classmethod
    def new(
        cls,
        optimal_costs: np.ndarray,
        previous_change_points: np.ndarray,
        pruning_fraction: float,
        interval_starts: np.ndarray,
        interval_ends: np.ndarray,
        interval_costs: np.ndarray,
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
            interval_starts=interval_starts,
            interval_ends=interval_ends,
            interval_costs=interval_costs,
        )


@njit(cache=True)
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
    cache: dict | None = None,
    log_costs: bool = False,
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
    log_costs : bool, optional
        If True, populate ``PELTResult.interval_starts``/``interval_ends``/
        ``interval_costs`` with every ``(start, end, cost)`` triple the DP
        evaluated. Off by default to avoid the O(n_evals) memory overhead
        when only the changepoints are needed.

    Returns
    -------
    PELTResult
        Container for the results of the PELT algorithm run.
    """
    check_is_fitted(cost)
    if cache is None:
        cache = cost.precompute(X)
    n_samples = X.shape[0]

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

    # Log of every (start, end) interval the DP evaluated, with its cost.
    # Only populated when ``log_costs`` is True.
    eval_starts_log: list[np.ndarray] = []
    eval_ends_log: list[np.ndarray] = []
    eval_costs_log: list[np.ndarray] = []
    if log_costs:
        eval_starts_log.append(non_changepoint_starts)
        eval_ends_log.append(non_changepoint_ends)
        eval_costs_log.append(non_changepoint_costs)

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
            if starts_to_prune.size:
                # cost_eval_starts and starts_to_prune are both sorted and unique.
                keep_mask = np.isin(
                    cost_eval_starts,
                    starts_to_prune,
                    assume_unique=True,
                    invert=True,
                )
                cost_eval_starts = cost_eval_starts[keep_mask]

        # Add the next start to the admissible starts set:
        cost_eval_starts = np.concatenate((cost_eval_starts, np.array([latest_start])))
        cost_eval_intervals = np.empty((cost_eval_starts.size, 2), dtype=np.int64)
        cost_eval_intervals[:, 0] = cost_eval_starts
        cost_eval_intervals[:, 1] = current_obs_ind + 1
        with skip_validation():
            interval_costs = np.sum(cost.evaluate(cache, cost_eval_intervals), axis=1)

        if log_costs:
            eval_starts_log.append(cost_eval_starts.copy())
            eval_ends_log.append(cost_eval_intervals[:, 1].copy())
            eval_costs_log.append(interval_costs)

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
        interval_starts=(
            np.concatenate(eval_starts_log)
            if eval_starts_log
            else np.empty(0, dtype=np.int64)
        ),
        interval_ends=(
            np.concatenate(eval_ends_log)
            if eval_ends_log
            else np.empty(0, dtype=np.int64)
        ),
        interval_costs=(
            np.concatenate(eval_costs_log) if eval_costs_log else np.empty(0)
        ),
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
    cache: dict | None = None,
    log_costs: bool = False,
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
    log_costs : bool, optional
        If True, populate ``PELTResult.interval_starts``/``interval_ends``/
        ``interval_costs`` with every ``(start, end, cost)`` triple the DP
        evaluated. Off by default to avoid the O(n_evals) memory overhead
        when only the changepoints are needed.

    Returns
    -------
    PELTResult
        Container for the results of the PELT algorithm run.
    """
    check_is_fitted(cost)
    if cache is None:
        cache = cost.precompute(X)
    n_samples = X.shape[0]
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

    # Log of every (start, end) interval the DP evaluated, with its cost.
    eval_starts_log: list[np.ndarray] = []
    eval_ends_log: list[np.ndarray] = []
    eval_costs_log: list[np.ndarray] = []

    for obs_interval_start, obs_interval_end in observation_intervals:
        # Add the next start to the admissible starts set:
        eval_starts = np.concatenate((eval_starts, np.array([obs_interval_start])))
        eval_ends = np.repeat(obs_interval_end + 1, len(eval_starts))
        eval_intervals = np.column_stack((eval_starts, eval_ends))
        interval_costs = np.sum(cost.evaluate(cache, eval_intervals), axis=1)

        if log_costs:
            eval_starts_log.append(eval_starts.copy())
            eval_ends_log.append(eval_ends)
            eval_costs_log.append(interval_costs)

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
        interval_starts=(
            np.concatenate(eval_starts_log)
            if eval_starts_log
            else np.empty(0, dtype=np.int64)
        ),
        interval_ends=(
            np.concatenate(eval_ends_log)
            if eval_ends_log
            else np.empty(0, dtype=np.int64)
        ),
        interval_costs=(
            np.concatenate(eval_costs_log) if eval_costs_log else np.empty(0)
        ),
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
        instance with ``score_type='cost'``.
        If ``None``, defaults to ``L2Cost()``.
    penalty : float or None, default=None
        Penalty incurred for each added changepoint. Must be non-negative.
        If ``None``, defaults to ``cost_.get_default_penalty()`` after fitting
        (BIC-based penalty). To sweep penalties for tuning, sweep this
        parameter directly (e.g. ``GridSearchCV`` over a log-spaced grid).
    min_segment_length : int or None, default=None
        Minimum number of samples in a segment. Must be at least ``cost.min_size``.
        If ``None``, defaults to ``2 * cost.min_size`` after fitting. The 2x
        factor provides a finite-sample safety floor that prevents spurious
        short segments from scale-estimating costs (e.g. Gaussian, Laplace).
    prune : bool, default=True
        If False, drop the pruning step. Reverts to optimal partitioning.
        Can be useful for debugging and testing.
    split_cost : float, default=0.0
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the above inequality.
    pruning_margin : float, default=0.0
        The pruning margin to use. By default set to zero.
        This is used to reduce pruning of the admissible starts set.
        Can be useful if the cost function is imprecise, i.e.
        based on solving an optimization problem with large tolerance.
    step_size : int, default=1
        Only indices that are multiples of ``step_size`` from the start are
        considered as potential changepoints. Implicitly ensures that
        ``min_segment_length >= step_size``, but it is an error to specify
        ``min_segment_length`` greater than ``step_size``.

        .. experimental::
            ``step_size`` is experimental and the parameter or its semantics
            may change in a future release.

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
    >>> detector.fit(X).predict(X)
    array([100])
    """

    _parameter_constraints = {
        "cost": [HasMethods(["fit", "precompute", "evaluate"]), None],
        "penalty": [Interval(Real, 0, None, closed="left"), None],
        "penalty_scale": [Interval(Real, 0, None, closed="neither")],
        "min_segment_length": [Interval(Integral, 1, None, closed="left"), None],
        "prune": ["boolean"],
        "split_cost": [Interval(Real, 0, None, closed="left")],
        "pruning_margin": [Interval(Real, 0, None, closed="left")],
        "step_size": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        cost: BaseCost | None = None,
        penalty: float | None = None,
        penalty_scale: float = 1.0,
        min_segment_length: int | None = None,
        prune: bool = True,
        split_cost: float = 0.0,
        pruning_margin: float = 0.0,
        step_size: int = 1,
    ):
        self.cost = cost
        self.penalty = penalty
        self.penalty_scale = penalty_scale
        self.min_segment_length = min_segment_length
        self.prune = prune
        self.split_cost = split_cost
        self.pruning_margin = pruning_margin
        self.step_size = step_size

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get tags, propagating input constraints from the cost."""
        tags = super().__sklearn_tags__()
        scorer_tags = _resolve_cost(self.cost).__sklearn_tags__()
        tags.input_tags = scorer_tags.input_tags
        tags.change_detector_tags.linear_trend_segment = (
            scorer_tags.interval_scorer_tags.linear_trend_segment
        )
        tags.change_detector_tags.calibration_strategy = "path_search"
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

        cost = _resolve_cost(self.cost)
        check_interval_scorer(
            cost,
            ensure_score_type=["cost"],
            allow_penalised=False,
            caller_name=self.__class__.__name__,
            arg_name="cost",
        )
        self.cost_ = clone(cost).fit(X, y)

        min_segment_length = (
            2 * self.cost_.min_size
            if self.min_segment_length is None
            else self.min_segment_length
        )
        if min_segment_length < self.cost_.min_size:
            raise ValueError(
                f"`min_segment_length` (={min_segment_length}) must be at least "
                f"`cost.min_size` (={self.cost_.min_size})."
            )
        if self.step_size > 1 and min_segment_length > self.step_size:
            raise ValueError(
                f"`min_segment_length` (={min_segment_length}) cannot be "
                f"greater than `step_size` (={self.step_size}) when step_size > 1."
            )
        self.min_segment_length_ = min_segment_length

        base_penalty = (
            self.cost_.get_default_penalty() if self.penalty is None else self.penalty
        )
        self.penalty_ = self.penalty_scale * base_penalty

        return self

    def _run(self, X: np.ndarray, log_costs: bool = False) -> PELTResult:
        """Run the appropriate PELT variant on ``X`` with the fitted state."""
        if self.step_size > 1:
            return _run_pelt_with_step_size(
                cost=self.cost_,
                X=X,
                penalty=self.penalty_,
                step_size=self.step_size,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
                log_costs=log_costs,
            )
        return _run_pelt(
            cost=self.cost_,
            X=X,
            penalty=self.penalty_,
            min_segment_length=self.min_segment_length_,
            split_cost=self.split_cost,
            prune=self.prune,
            pruning_margin=self.pruning_margin,
            log_costs=log_costs,
        )

    def predict_all(self, X: ArrayLike) -> dict:
        """Run PELT and return all outputs in a single pass.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyse for changepoints.

        Returns
        -------
        result : dict with keys:

            ``"changepoints"`` : np.ndarray of shape (n_changepoints,)
                Sorted integer indices of detected changepoints.
            ``"cumulative_optimal_costs"`` : np.ndarray of shape (n_samples,)
                Cumulative optimal costs.
            ``"previous_changepoints"`` : np.ndarray of shape (n_samples,)
                For each sample, the start of the optimal segment ending there.
            ``"pruning_fraction"`` : float
                Fraction of candidate starts pruned vs. optimal partitioning.
            ``"interval_starts"`` : np.ndarray
                Start indices of all ``(start, end)`` intervals the DP
                evaluated.
            ``"interval_ends"`` : np.ndarray
                End indices of all intervals the DP evaluated.
            ``"interval_costs"`` : np.ndarray
                Unpenalised (feature-summed) cost at each evaluated interval.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True)

        pelt_result = self._run(X, log_costs=True)
        return {
            "changepoints": pelt_result.changepoints.astype(np.intp),
            "cumulative_optimal_costs": pelt_result.optimal_costs,
            "previous_changepoints": pelt_result.previous_change_points,
            "pruning_fraction": pelt_result.pruning_fraction,
            "interval_starts": pelt_result.interval_starts,
            "interval_ends": pelt_result.interval_ends,
            "interval_costs": pelt_result.interval_costs,
        }

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Detect changepoints in a time series.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyse for changepoints.

        Returns
        -------
        changepoints : np.ndarray of shape (n_changepoints,)
            Sorted integer indices of detected changepoints. A changepoint is defined
            as the first index of a segment, such that the data segments are given
            by ``X[:cpt[0]], X[cpt[0]:cpt[1]], ..., X[cpt[-1]:]``.
            Empty array if no changepoints are detected.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True)
        pelt_result = self._run(X, log_costs=False)
        return pelt_result.changepoints.astype(np.intp)

    def predict_scores(
        self,
        X: ArrayLike,
        return_index: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return the cost at every interval the PELT dynamic programme evaluated.

        Runs PELT on ``X`` with the fitted state and returns the
        feature-summed cost value at every ``(start, end)`` interval the
        dynamic programme actually evaluated. With pruning enabled, the set
        of evaluated intervals depends on the current ``penalty_``; set
        ``prune=False`` (or run via :func:`skchange.new_api.tuning.unpenalised_scores`
        with the penalty zeroed) to obtain the full optimal-partitioning grid.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to evaluate.
        return_index : bool, default=False
            If ``True``, also return a dict locating each score on the time
            axis. See the Returns section for the keys.

        Returns
        -------
        scores : np.ndarray of shape (n_intervals,)
            Unpenalised cost values, one per evaluated interval. Returned
            alone when ``return_index=False``.
        index : dict, optional
            Only returned when ``return_index=True``. Contains:

            - ``"starts"`` : np.ndarray of shape (n_intervals,)
              Start indices of the evaluated intervals.
            - ``"ends"`` : np.ndarray of shape (n_intervals,)
              End indices of the evaluated intervals.
        """
        result = self.predict_all(X)
        scores = result["interval_costs"]
        if return_index:
            return scores, {
                "starts": result["interval_starts"],
                "ends": result["interval_ends"],
            }
        return scores
