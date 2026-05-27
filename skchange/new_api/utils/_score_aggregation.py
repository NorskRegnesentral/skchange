"""Aggregation and penalisation of interval scorer outputs.

Detectors keep a fitted unpenalised scorer and convert its feature-wise raw
scores into a 1D aggregated, penalised score array via
:func:`aggregate_and_penalise`. Aggregation parameters are resolved at fit
time via :func:`resolve_aggregation`.

Five internal modes are supported:

- ``"sum"``: ``sum(scores, axis=1) - penalty`` (scalar penalty).
- ``"max"``: ``max(scores, axis=1) - penalty`` (scalar penalty).
- ``"top_k_linear"``: top-k aggregation with a linear non-decreasing
  penalty array.
- ``"top_k_nonlinear"``: top-k aggregation with a general non-decreasing
  penalty array.
- ``"passthrough"``: scorer is inherently penalised; raw scores are returned
  as a 1D array unchanged.
"""

__all__ = [
    "AGG_MODES",
    "USER_AGG_CHOICES",
    "aggregate_and_penalise",
    "resolve_aggregation",
    "resolve_penalty",
]

import warnings

import numpy as np

from skchange.new_api.interval_scorers._base import (
    BaseIntervalScorer,
    is_penalised_score,
)
from skchange.new_api.utils._numba import njit
from skchange.new_api.utils.validation import check_penalty

# User-facing aggregation choices for detectors' ``agg`` parameter.
USER_AGG_CHOICES = ("sum", "max")

# Internal aggregation modes returned by ``resolve_aggregation`` and consumed
# by ``aggregate_and_penalise``.
AGG_MODES = ("sum", "max", "top_k_linear", "top_k_nonlinear", "passthrough")


@njit(cache=True)
def _agg_sum(scores: np.ndarray, penalty: float) -> np.ndarray:
    """Sum over features minus a scalar penalty."""
    return scores.sum(axis=1) - penalty


@njit(cache=True)
def _agg_max(scores: np.ndarray, penalty: float) -> np.ndarray:
    """Max over features minus a scalar penalty."""
    n = scores.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = scores[i].max() - penalty
    return out


@njit(cache=True)
def _top_k_linear(scores: np.ndarray, penalty_values: np.ndarray) -> np.ndarray:
    """Top-k aggregation with a linear non-decreasing penalty."""
    penalty_slope = penalty_values[1] - penalty_values[0]
    penalty_intercept = penalty_values[0] - penalty_slope
    penalised = np.maximum(scores - penalty_slope, 0.0) - penalty_intercept
    return penalised.sum(axis=1)


@njit(cache=True)
def _top_k_nonlinear(scores: np.ndarray, penalty_values: np.ndarray) -> np.ndarray:
    """Top-k aggregation with a general non-decreasing penalty."""
    n = scores.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        sorted_scores = np.sort(scores[i])[::-1]
        out[i] = np.max(np.cumsum(sorted_scores) - penalty_values)
    return out


def resolve_penalty(
    scorer: BaseIntervalScorer,
    penalty: float | np.ndarray | None,
    penalty_scale: float,
    *,
    caller_name: str,
    scorer_param_name: str = "change_score",
) -> float | np.ndarray | None:
    """Resolve the effective penalty for a fitted scorer.

    For inherently penalised scorers (tag ``penalised=True``) returns
    ``None`` and warns if the user supplied ``penalty`` or a non-default
    ``penalty_scale``. Otherwise falls back to
    ``scorer.get_default_penalty()`` when ``penalty is None``, applies
    ``penalty_scale``, and validates via :func:`check_penalty`.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        Fitted scorer.
    penalty : float, np.ndarray or None
        User-supplied penalty.
    penalty_scale : float
        Multiplicative factor applied to the resolved penalty.
    caller_name : str
        Name of the calling class, used in error / warning messages.
    scorer_param_name : str, default="change_score"
        Name of the scorer parameter on the calling estimator. Used to make
        the passthrough warning accurate for detectors whose scorer is
        called something other than ``change_score`` (e.g. ``point_saving``).

    Returns
    -------
    effective_penalty : float, np.ndarray or None
        ``None`` when ``scorer`` is inherently penalised; otherwise the
        resolved base penalty (``penalty`` if not ``None``, else
        ``scorer.get_default_penalty()``) multiplied by ``penalty_scale``
        and validated by :func:`check_penalty`. A Python float for scalar
        penalties, a 1D float64 ndarray for array penalties.
    """
    if is_penalised_score(scorer):
        if penalty is not None or penalty_scale != 1.0:
            warnings.warn(
                f"`{caller_name}.penalty` and `penalty_scale` are ignored "
                f"when `{scorer_param_name}` is already a penalised scorer; "
                "the penalty owned by the supplied scorer is used instead.",
                UserWarning,
                stacklevel=3,
            )
        return None

    base_penalty = scorer.get_default_penalty() if penalty is None else penalty
    return check_penalty(
        np.asarray(base_penalty) * penalty_scale,
        caller_name=caller_name,
        arg_name="penalty",
    )


def resolve_aggregation(
    scorer: BaseIntervalScorer,
    agg: str,
    penalty: float | np.ndarray | None,
    n_features: int,
    *,
    caller_name: str | None = None,
    scorer_param_name: str = "change_score",
) -> str:
    """Classify ``(scorer, agg, penalty)`` into an aggregation mode.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        Fitted scorer. If it is inherently penalised (tag ``penalised=True``)
        the mode is ``"passthrough"``.
    agg : {"sum", "max"}
        User-supplied aggregation. Only consulted when ``penalty`` is scalar.
        With an array-valued ``penalty`` the mode is forced to
        ``"top_k_linear"``/``"top_k_nonlinear"`` and ``agg`` must be
        ``"sum"`` (the default), otherwise a ``ValueError`` is raised.
    penalty : float, np.ndarray or None
        Already-validated penalty (e.g. from
        :func:`skchange.new_api.utils.validation.check_penalty`). ``None``
        signals that ``scorer`` is inherently penalised; in that case it must
        match ``is_penalised_score(scorer)``.
    n_features : int
        Number of features in the input data. Used to validate the length of
        array penalties.
    caller_name : str, optional
        Name of the calling class, used in warning messages. When ``None``,
        the passthrough warning for a non-default ``agg`` is suppressed.
    scorer_param_name : str, default="change_score"
        Name of the scorer parameter on the calling estimator. Used to make
        the passthrough warning accurate for detectors whose scorer is
        called something other than ``change_score``.

    Returns
    -------
    mode : str
        One of :data:`AGG_MODES`.
    """
    if is_penalised_score(scorer):
        if agg != "sum" and caller_name is not None:
            warnings.warn(
                f"`{caller_name}.agg` is ignored when `{scorer_param_name}` "
                "is already a penalised scorer; the scorer's own aggregation "
                "is used instead.",
                UserWarning,
                stacklevel=3,
            )
        return "passthrough"

    scorer_tags = scorer.__sklearn_tags__().interval_scorer_tags
    is_array = isinstance(penalty, np.ndarray)

    if scorer_tags.aggregated and is_array:
        raise ValueError("`penalty` must be scalar for aggregated input scores.")
    if is_array and penalty.size != n_features:
        raise ValueError(
            "`penalty` must be scalar or have length equal to n_features. "
            f"Got penalty length {penalty.size} and n_features {n_features}."
        )

    if not is_array:
        return agg

    # Array penalty implies top-k aggregation. Only consistent with the
    # default ``agg="sum"``; any other choice is a user error.
    if agg != "sum":
        raise ValueError(
            f"`agg={agg!r}` is incompatible with an array-valued `penalty`. "
            "Array penalties imply top-k aggregation; use `agg='sum'` or "
            "pass a scalar penalty."
        )
    diffs = np.diff(penalty)
    return "top_k_linear" if np.allclose(diffs, diffs[0]) else "top_k_nonlinear"


def aggregate_and_penalise(
    raw_scores: np.ndarray,
    mode: str,
    penalty: float | np.ndarray | None,
) -> np.ndarray:
    """Aggregate feature-wise raw scores and apply a penalty.

    Parameters
    ----------
    raw_scores : np.ndarray of shape (n_intervals, n_features)
        Scores returned by ``BaseIntervalScorer.evaluate``. Must be 2D; for
        inherently penalised / aggregated scorers ``n_features == 1``.
    mode : str
        One of :data:`AGG_MODES`, as produced by :func:`resolve_aggregation`.
    penalty : float, np.ndarray or None
        Effective penalty produced by :func:`resolve_aggregation`. Must be
        ``None`` for ``mode="passthrough"`` and non-``None`` otherwise.

    Returns
    -------
    out : np.ndarray of shape (n_intervals,)
        1D aggregated, penalised scores.
    """
    if mode == "passthrough":
        return raw_scores.reshape(-1)
    if mode == "sum":
        return _agg_sum(raw_scores, penalty)
    if mode == "max":
        return _agg_max(raw_scores, penalty)
    if mode == "top_k_linear":
        return _top_k_linear(raw_scores, penalty)
    if mode == "top_k_nonlinear":
        return _top_k_nonlinear(raw_scores, penalty)
    raise ValueError(f"Unknown aggregation mode: {mode!r}.")
