"""Functions to validate inputs and parameters in skchange."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import clone as sklearn_clone

from skchange.new_api.typing import ArrayLike

if TYPE_CHECKING:
    from skchange.new_api.interval_scorers._base import BaseIntervalScorer


def check_interval_scorer(
    scorer: BaseIntervalScorer,
    required_tasks: list | None = None,
    allow_penalised: bool = True,
    clone: bool = True,
    caller_name: str | None = None,
    arg_name: str = "",
) -> BaseIntervalScorer:
    """Check if the given scorer is a valid interval scorer.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        The scorer to check.
    required_tasks : list of str or None, default=None
        If specified, the scorer's score_type tag must be one of these.
    allow_penalised : bool, default=True
        Whether to allow penalised scorers. If False, raises error if scorer has
        penalised tag True.
    clone: bool, default=True
        Whether to clone the scorer before returning.
    caller_name : str or None, default=None
        Name of the caller for error messages.
    arg_name : str, default=""
        Name of the argument for error messages.

    Returns
    -------
    BaseIntervalScorer
        The validated input scorer.

    """
    if clone:
        scorer = sklearn_clone(scorer)

    score_type = scorer.__sklearn_tags__().interval_scorer_tags.score_type
    if required_tasks and score_type not in required_tasks:
        _required_tasks = [f'"{task}"' for task in required_tasks]
        tasks_str = (
            ", ".join(_required_tasks[:-1]) + " or " + _required_tasks[-1]
            if len(_required_tasks) > 1
            else _required_tasks[0]
        )
        raise ValueError(
            f"{caller_name} requires `{arg_name}` to have score_type {tasks_str}"
            f" ({arg_name}.__sklearn_tags__().interval_scorer_tags.score_type "
            f"in {required_tasks}). "
            f'Got {scorer.__class__.__name__}, which has score_type "{score_type}".'
        )
    if not allow_penalised and scorer.__sklearn_tags__().interval_scorer_tags.penalised:
        raise ValueError(f"`{arg_name}` cannot be a penalised score.")

    return scorer


def check_penalty(
    penalty: float | ArrayLike,
    ensure_non_decreasing: bool = True,
    copy: bool = True,
    caller_name: str | None = None,
    arg_name: str = "",
) -> float | np.ndarray:
    """Check if the given penalty is valid.

    Parameters
    ----------
    penalty : float | ArrayLike
        The penalty to check.
    ensure_non_decreasing : bool, default=True
        If True, the penalty must be non-decreasing.
    copy : bool, default=True
        Whether to copy the penalty array. Ignored if penalty is a scalar.
    caller_name : str | None, default=None
        The name of the caller. Used for error messages.
    arg_name : str
        The name of the argument. Used for error messages.
    """
    from sklearn.utils.validation import check_array

    penalty = np.asarray(penalty).squeeze()
    if penalty.ndim == 0:
        penalty = penalty.reshape(1)

    penalty = check_array(
        penalty,
        ensure_2d=False,  # penalty should be 1D after squeezing.
        dtype=np.float64,
        copy=copy,
        ensure_all_finite=True,
    )

    if penalty.ndim != 1:
        raise ValueError(
            f"`{arg_name}` must be a 1D array in {caller_name}."
            f" Got {penalty.ndim}D array."
        )

    if not np.all(penalty >= 0.0):
        raise ValueError(f"`{arg_name}` must be non-negative in {caller_name}")

    if ensure_non_decreasing and penalty.size > 1 and np.any(np.diff(penalty) < 0):
        raise ValueError(f"`{arg_name}` must be non-decreasing in {caller_name}")

    if penalty.size == 1:
        return float(penalty[0])

    return penalty
