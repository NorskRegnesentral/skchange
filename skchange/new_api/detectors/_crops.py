"""CROPS algorithm for path solutions to the PELT algorithm."""

__author__ = ["johannvk", "Tveten"]
__all__ = ["CROPS"]

import warnings
from numbers import Integral, Real

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.detectors._base import BaseChangeDetector
from skchange.new_api.detectors._pelt import (
    _run_pelt,
    _run_pelt_min_segment_length_one,
    _run_pelt_with_step_size,
)
from skchange.new_api.interval_scorers._base import BaseCost
from skchange.new_api.interval_scorers._change_scores.continuous_linear_trend_score import (  # noqa: E501
    _lin_reg_cont_piecewise_linear_trend_score,
)
from skchange.new_api.interval_scorers._costs.l2_cost import L2Cost
from skchange.new_api.typing import ArrayLike, Self
from skchange.new_api.utils import SkchangeTags
from skchange.new_api.utils._param_validation import (
    HasMethods,
    Interval,
    StrOptions,
    _fit_context,
)
from skchange.new_api.utils.validation import check_interval_scorer, validate_data


def _resolve_cost(cost: BaseCost | None) -> BaseCost:
    """Return cost or the default L2Cost()."""
    return cost if cost is not None else L2Cost()


def _evaluate_segmentation(
    cost: BaseCost,
    cache: dict,
    changepoints: np.ndarray,
    n_samples: int,
) -> float:
    """Evaluate the total cost of a segmentation defined by ``changepoints``.

    Each entry in ``changepoints`` is the first index of a new segment.
    """
    if changepoints.ndim != 1:
        changepoints = changepoints.reshape(-1)

    if len(changepoints) == 0:
        boundaries = np.array([0, n_samples], dtype=np.intp)
    else:
        if np.any(np.diff(changepoints) <= 0):
            raise ValueError("`changepoints` must contain strictly increasing entries.")
        boundaries = np.concatenate(
            (
                np.array([0], dtype=np.intp),
                changepoints.astype(np.intp),
                np.array([n_samples], dtype=np.intp),
            )
        )

    intervals = np.column_stack((boundaries[:-1], boundaries[1:]))
    return float(np.sum(cost.evaluate(cache, intervals)))


def _segmentation_bic_value(
    cost: BaseCost,
    cache: dict,
    changepoints: np.ndarray,
    n_samples: int,
) -> float:
    """BIC value for a segmentation: ``segmentation_cost + n_segments * bic_penalty``.

    Uses ``cost.get_default_penalty()`` as the per-segment model-complexity term,
    which encodes the cost-specific number of parameters per segment. This is
    independent of the optimisation penalty and used purely for selecting among
    candidate segmentations.
    """
    n_segments = len(changepoints) + 1
    seg_cost = _evaluate_segmentation(cost, cache, changepoints, n_samples)
    per_segment_penalty = float(np.sum(cost.get_default_penalty()))
    return seg_cost + n_segments * per_segment_penalty


def _crops_elbow_scores(
    num_changepoints: np.ndarray,
    optimum_value: np.ndarray,
) -> np.ndarray:
    """Calculate the elbow scores for a CROPS solution path.

    The elbow score at each intermediate number of changepoints quantifies the
    evidence for a change in slope of ``optimum_value`` regressed on
    ``num_changepoints``. Larger is better. The first and last entries are
    ``-inf`` since a slope change cannot be evaluated there.
    """
    n = len(num_changepoints)
    if n < 3:
        warnings.warn(
            f"Not enough segmentations ({n}) to calculate the elbow cost. "
            "Returning -inf for all segmentations."
        )
        return np.full(n, -np.inf)

    elbow_values = _lin_reg_cont_piecewise_linear_trend_score(
        starts=np.repeat(0, n - 2),
        splits=np.arange(1, n - 1),
        ends=np.repeat(n, n - 2),
        X=optimum_value.reshape(-1, 1),
        times=num_changepoints,
    )
    return np.concatenate(
        (np.array([-np.inf]), elbow_values.reshape(-1), np.array([-np.inf]))
    )


def _format_crops_results(
    penalty_to_solution_dict: dict[float, tuple[np.ndarray, float]],
    penalty_nudge: float,
) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
    """Format the raw CROPS solution dict into a metadata table and lookup.

    Returns
    -------
    metadata : dict of np.ndarray
        Dictionary with keys ``num_changepoints``, ``penalty``,
        ``segmentation_cost``, and ``optimum_value``. Rows are sorted by
        ``num_changepoints`` ascending.
    changepoints_lookup : dict[int, np.ndarray]
        Mapping from number of changepoints to the corresponding indices.
    """
    list_results = [
        (len(changepoints), penalty, segmentation_cost, changepoints)
        for (
            penalty,
            (changepoints, segmentation_cost),
        ) in penalty_to_solution_dict.items()
    ]

    # Sort by num_changepoints desc, then by penalty asc (so we keep the lowest
    # penalty for each num_changepoints after deduplication below).
    list_results.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    # Replace each penalty by the critical penalty separating it from the
    # adjacent solution with one fewer changepoint, when applicable.
    for i in range(len(list_results) - 1):
        n_cp_low = list_results[i][0]
        n_cp_high = list_results[i + 1][0]
        if n_cp_low == n_cp_high + 1:
            critical_penalty = (list_results[i + 1][2] - list_results[i][2]) * (
                1.0 + penalty_nudge
            )
            list_results[i + 1] = (
                list_results[i + 1][0],
                critical_penalty,
                list_results[i + 1][2],
                list_results[i + 1][3],
            )

    # Deduplicate by num_changepoints (keep the first occurrence).
    encountered = set()
    unique_results = []
    for entry in list_results:
        n_cp = entry[0]
        if n_cp in encountered:
            continue
        encountered.add(n_cp)
        unique_results.append(entry)

    changepoints_lookup = {entry[0]: entry[3] for entry in unique_results}

    # Reverse so num_changepoints is ascending.
    unique_results = unique_results[::-1]
    num_cp = np.array([e[0] for e in unique_results], dtype=np.intp)
    penalty = np.array([e[1] for e in unique_results], dtype=float)
    seg_cost = np.array([e[2] for e in unique_results], dtype=float)
    optimum_value = seg_cost + penalty * num_cp
    metadata = {
        "num_changepoints": num_cp,
        "penalty": penalty,
        "segmentation_cost": seg_cost,
        "optimum_value": optimum_value,
    }
    return metadata, changepoints_lookup


def _solve_for_changepoints(
    cost: BaseCost,
    X: np.ndarray,
    penalty: float,
    min_segment_length: int,
    step_size: int,
    split_cost: float,
    prune: bool,
    pruning_margin: float,
) -> np.ndarray:
    """Run the appropriate PELT variant for a single penalty."""
    if step_size > 1:
        result = _run_pelt_with_step_size(
            cost=cost,
            X=X,
            penalty=penalty,
            step_size=step_size,
            split_cost=split_cost,
            prune=prune,
            pruning_margin=pruning_margin,
        )
    elif min_segment_length == 1:
        result = _run_pelt_min_segment_length_one(
            cost=cost,
            X=X,
            penalty=penalty,
            split_cost=split_cost,
            prune=prune,
            pruning_margin=pruning_margin,
        )
    else:
        result = _run_pelt(
            cost=cost,
            X=X,
            penalty=penalty,
            min_segment_length=min_segment_length,
            split_cost=split_cost,
            prune=prune,
            pruning_margin=pruning_margin,
        )
    return result.changepoints


def _run_crops(
    cost: BaseCost,
    X: np.ndarray,
    min_penalty: float,
    max_penalty: float,
    min_segment_length: int,
    step_size: int,
    split_cost: float,
    prune: bool,
    pruning_margin: float,
    middle_penalty_nudge: float,
) -> dict[float, tuple[np.ndarray, float]]:
    """Run the CROPS path-search algorithm.

    Returns a dictionary mapping each evaluated penalty to a
    ``(changepoints, segmentation_cost)`` tuple.
    """
    cache = cost.precompute(X)
    n_samples = X.shape[0]

    def _solve(penalty: float) -> tuple[np.ndarray, float]:
        cps = _solve_for_changepoints(
            cost=cost,
            X=X,
            penalty=penalty,
            min_segment_length=min_segment_length,
            step_size=step_size,
            split_cost=split_cost,
            prune=prune,
            pruning_margin=pruning_margin,
        )
        seg_cost = _evaluate_segmentation(cost, cache, cps, n_samples)
        return cps, seg_cost

    min_cps, min_seg_cost = _solve(min_penalty)
    max_cps, max_seg_cost = _solve(max_penalty)

    penalty_to_solution: dict[float, tuple[np.ndarray, float]] = {
        min_penalty: (min_cps, min_seg_cost),
        max_penalty: (max_cps, max_seg_cost),
    }

    intervals_to_search: list[tuple[float, float]] = []
    if len(min_cps) > len(max_cps) + 1:
        intervals_to_search.append((min_penalty, max_penalty))

    while intervals_to_search:
        low_pen, high_pen = intervals_to_search.pop(0)
        low_cps, low_seg_cost = penalty_to_solution[low_pen]
        high_cps, high_seg_cost = penalty_to_solution[high_pen]
        n_low, n_high = len(low_cps), len(high_cps)

        if n_low <= n_high + 1:
            continue

        threshold_pen = (high_seg_cost - low_seg_cost) / (n_low - n_high)
        additive_nudge = (
            threshold_pen + (high_pen - threshold_pen) * middle_penalty_nudge
        )
        multiplicative_nudge = threshold_pen * (1.0 + middle_penalty_nudge)
        middle_pen = min(additive_nudge, multiplicative_nudge)

        middle_cps, middle_seg_cost = _solve(middle_pen)
        penalty_to_solution[middle_pen] = (middle_cps, middle_seg_cost)
        n_mid = len(middle_cps)

        if n_mid == n_high:
            continue
        elif n_mid == n_low:
            # Numerical imprecision in the cost (or aggressive pruning) prevented
            # PELT from finding a finer-grained solution at the nudged middle
            # penalty. Skip further refinement of this sub-interval; the path is
            # incomplete but still valid for selection. Increase
            # `middle_penalty_nudge`, `pruning_margin`, or `split_cost` if a
            # complete path is required.
            warnings.warn(
                "CROPS could not refine the penalty interval "
                f"[{low_pen}, {high_pen}] further: the middle penalty produced "
                "the same number of changepoints as the low penalty. The "
                "returned solution path may be incomplete.",
                stacklevel=2,
            )
            continue
        else:
            intervals_to_search.append((low_pen, middle_pen))
            intervals_to_search.append((middle_pen, high_pen))
            intervals_to_search.sort(key=lambda x: x[0])

    return penalty_to_solution


class CROPS(BaseChangeDetector):
    """CROPS algorithm for path solutions to the PELT algorithm.

    This change detector solves for all penalised optimal partitionings within
    the penalty range ``[min_penalty, max_penalty]`` using the CROPS algorithm
    [1]_, which in turn employs the PELT algorithm to repeatedly solve
    penalised optimal partitioning problems for different penalties.

    When predicting changepoints through ``predict_changepoints`` (or
    ``predict_all``), the detector selects the best segmentation among the
    optimal partitionings within the penalty range, using the
    ``selection_method`` criterion.

    Parameters
    ----------
    cost : BaseCost or None, default=None
        The cost function to use. Must be a ``BaseCost`` instance with
        ``score_type='cost'``. Passing a ``PenalisedScore`` will raise a
        ``ValueError`` in ``fit``. If ``None``, defaults to ``L2Cost()``.
    min_penalty : float or None, default=None
        The start of the penalty solution interval. Must be non-negative and
        strictly less than ``max_penalty``. If ``None``, defaults to
        ``0.5 * cost.get_default_penalty()`` after fitting.
    max_penalty : float or None, default=None
        The end of the penalty solution interval. Must be strictly greater
        than ``min_penalty``. If ``None``, defaults to
        ``5.0 * cost.get_default_penalty()`` after fitting.
    selection_method : str, default="bic"
        The segmentation selection method to use when selecting the best
        segmentation among the optimal segmentations found within the penalty
        range ``[min_penalty, max_penalty]``. Options:

        * ``"bic"``: Select the segmentation with the lowest Bayesian
          Information Criterion, computed as
          ``segmentation_cost + n_segments * cost.get_default_penalty()``.
        * ``"elbow"``: Select the segmentation with the highest elbow score,
          defined as the improvement in squared residuals when allowing a
          change in slope of segmentation cost regressed on the number of
          changepoints, evaluated at each intermediate number of changepoints.
    min_segment_length : int or None, default=None
        Minimum number of samples in a segment. Must be at least
        ``cost.min_size``. If ``None``, defaults to ``cost.min_size`` after
        fitting.
    step_size : int, default=1
        Only indices that are multiples of ``step_size`` from the start are
        considered as potential changepoints. Implicitly ensures that
        ``min_segment_length >= step_size``, but it is an error to specify
        ``min_segment_length`` greater than ``step_size`` when ``step_size > 1``.
    split_cost : float, default=0.0
        The cost of splitting a segment, to ensure that
        ``cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)])``,
        for all valid splits ``0 <= t < p < s <= len(X) - 1``. The default of
        ``0.0`` is sufficient for log-likelihood cost functions.
    prune : bool, default=True
        If ``False``, drop the pruning step, reverting to optimal partitioning.
        Useful for debugging and testing.
    pruning_margin : float, default=0.0
        The pruning margin to use. Used to reduce pruning of the admissible
        starts set, which can be useful when the cost function is imprecise
        (e.g. based on solving an optimisation problem with large tolerance).
    middle_penalty_nudge : float, default=1e-5
        When computing the threshold penalty value separating PELT solutions
        with differing numbers of changepoints, the penalty is nudged upwards
        in order to solve for the segmentation with fewer changepoints. The
        default of ``1e-5`` is sufficient for most cases.

    Attributes
    ----------
    cost_ : BaseCost
        Fitted cost scorer.
    min_penalty_ : float
        Effective ``min_penalty`` used after resolving the default.
    max_penalty_ : float
        Effective ``max_penalty`` used after resolving the default.
    min_segment_length_ : int
        Effective minimum segment length used.

    References
    ----------
    .. [1] Haynes, K., Eckley, I. A., & Fearnhead, P. (2017). Computationally
       efficient changepoint detection for a range of penalties. Journal of
       Computational and Graphical Statistics, 26(1), 134-143.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.detectors import CROPS
    >>> rng = np.random.default_rng(2)
    >>> X = np.concatenate([rng.normal(0, 1, (100, 1)),
    ...                     rng.normal(10, 1, (100, 1))])
    >>> detector = CROPS(min_penalty=1.0, max_penalty=50.0)
    >>> detector.fit(X).predict_changepoints(X)
    array([100])
    """

    _parameter_constraints = {
        "cost": [HasMethods(["fit", "precompute", "evaluate"]), None],
        "min_penalty": [Interval(Real, 0, None, closed="left"), None],
        "max_penalty": [Interval(Real, 0, None, closed="left"), None],
        "selection_method": [StrOptions({"bic", "elbow"})],
        "min_segment_length": [Interval(Integral, 1, None, closed="left"), None],
        "step_size": [Interval(Integral, 1, None, closed="left")],
        "split_cost": [Interval(Real, 0, None, closed="left")],
        "prune": ["boolean"],
        "pruning_margin": [Interval(Real, 0, None, closed="left")],
        "middle_penalty_nudge": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        cost: BaseCost | None = None,
        min_penalty: float | None = None,
        max_penalty: float | None = None,
        selection_method: str = "bic",
        min_segment_length: int | None = None,
        step_size: int = 1,
        split_cost: float = 0.0,
        prune: bool = True,
        pruning_margin: float = 0.0,
        middle_penalty_nudge: float = 1e-5,
    ):
        self.cost = cost
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.selection_method = selection_method
        self.min_segment_length = min_segment_length
        self.step_size = step_size
        self.split_cost = split_cost
        self.prune = prune
        self.pruning_margin = pruning_margin
        self.middle_penalty_nudge = middle_penalty_nudge

    def __sklearn_tags__(self) -> SkchangeTags:
        """Get tags, propagating input constraints from the cost."""
        tags = super().__sklearn_tags__()
        scorer_tags = _resolve_cost(self.cost).__sklearn_tags__()
        tags.input_tags = scorer_tags.input_tags
        tags.change_detector_tags.linear_trend_segment = (
            scorer_tags.interval_scorer_tags.linear_trend_segment
        )
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
        self : CROPS
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

        default_penalty = float(np.sum(self.cost_.get_default_penalty()))
        self.min_penalty_ = (
            0.5 * default_penalty if self.min_penalty is None else self.min_penalty
        )
        self.max_penalty_ = (
            5.0 * default_penalty if self.max_penalty is None else self.max_penalty
        )
        if self.min_penalty_ >= self.max_penalty_:
            raise ValueError(
                f"`min_penalty` (={self.min_penalty_}) must be strictly less than "
                f"`max_penalty` (={self.max_penalty_})."
            )

        min_segment_length = (
            self.cost_.min_size
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

        return self

    def predict_all(self, X: ArrayLike) -> dict:
        """Run CROPS and return all outputs in a single pass.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyse for changepoints.

        Returns
        -------
        result : dict with keys:

            ``"changepoints"`` : np.ndarray of shape (n_changepoints,)
                Sorted integer indices of the selected changepoints.
            ``"changepoints_metadata"`` : dict of np.ndarray
                Dictionary with keys ``num_changepoints``, ``penalty``,
                ``segmentation_cost``, ``optimum_value``, and either
                ``bic_value`` or ``elbow_score``. All arrays have the same
                length and rows are aligned by index, sorted by
                ``num_changepoints`` ascending.
            ``"changepoints_lookup"`` : dict[int, np.ndarray]
                Mapping from number of changepoints to the corresponding
                changepoint indices for each segmentation in the path.
            ``"optimal_penalty"`` : float
                Penalty corresponding to the selected segmentation.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True)

        cost = clone(self.cost_).fit(X)
        cache = cost.precompute(X)
        n_samples = X.shape[0]

        penalty_to_solution = _run_crops(
            cost=cost,
            X=X,
            min_penalty=self.min_penalty_,
            max_penalty=self.max_penalty_,
            min_segment_length=self.min_segment_length_,
            step_size=self.step_size,
            split_cost=self.split_cost,
            prune=self.prune,
            pruning_margin=self.pruning_margin,
            middle_penalty_nudge=self.middle_penalty_nudge,
        )
        metadata, changepoints_lookup = _format_crops_results(
            penalty_to_solution_dict=penalty_to_solution,
            penalty_nudge=self.middle_penalty_nudge,
        )

        if self.selection_method == "elbow":
            optim = metadata["optimum_value"]
            shifted_optim = optim - optim.min()
            elbow_scores = _crops_elbow_scores(
                metadata["num_changepoints"], shifted_optim
            )
            metadata["elbow_score"] = elbow_scores
            best_idx = int(np.argmax(elbow_scores))
        else:  # "bic"
            bic_values = np.array(
                [
                    _segmentation_bic_value(
                        cost=cost,
                        cache=cache,
                        changepoints=changepoints_lookup[int(n_cp)],
                        n_samples=n_samples,
                    )
                    for n_cp in metadata["num_changepoints"]
                ]
            )
            metadata["bic_value"] = bic_values
            best_idx = int(np.argmin(bic_values))

        optimal_num_changepoints = int(metadata["num_changepoints"][best_idx])
        optimal_penalty = float(metadata["penalty"][best_idx])
        changepoints = changepoints_lookup[optimal_num_changepoints].astype(np.intp)

        return {
            "changepoints": changepoints,
            "changepoints_metadata": metadata,
            "changepoints_lookup": changepoints_lookup,
            "optimal_penalty": optimal_penalty,
        }

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
        return self.predict_all(X)["changepoints"]
