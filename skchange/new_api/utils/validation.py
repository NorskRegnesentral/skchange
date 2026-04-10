"""Functions to validate inputs and parameters in skchange."""

from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from sklearn.utils.validation import validate_data as _sklearn_validate_data

from skchange.new_api.typing import ArrayLike

if TYPE_CHECKING:
    from skchange.new_api.interval_scorers._base import BaseIntervalScorer


def validate_data(
    _estimator: BaseEstimator,
    /,
    X: ArrayLike,
    **kwargs,
) -> np.ndarray:
    """Validate X and set n_features_in_ and n_samples_in_ on the estimator.

    Thin wrapper around sklearn's ``validate_data`` that additionally stores
    the number of samples as ``_estimator.n_samples_in_`` when ``reset=True``
    (i.e. during fit), which is required for default penalty computation.

    Parameters
    ----------
    _estimator : BaseEstimator
        The estimator being fitted or applied. Modified in-place.
    X : array-like of shape (n_samples, n_features)
        Data to validate.
    **kwargs
        Forwarded to ``sklearn.utils.validation.validate_data``.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Validated array.
    """
    X = _sklearn_validate_data(_estimator, X, **kwargs)
    if kwargs.get("reset", True):
        _estimator.n_samples_in_ = X.shape[0]
    return X


def check_interval_specs(
    interval_specs: ArrayLike,
    n_cols: int,
    *,
    n_samples: int | None = None,
    check_sorted: bool = False,
    min_size: int | None = None,
    caller_name: str | None = None,
    arg_name: str = "interval_specs",
) -> np.ndarray:
    """Validate an interval_specs array.

    Always checks that the input is a 2D integer array with exactly ``n_cols``
    columns. Heavier checks are opt-in.

    Parameters
    ----------
    interval_specs : array-like of shape (n_interval_specs, n_cols)
        Interval specifications to validate.
    n_cols : int
        Required number of columns.
    n_samples : int or None, default=None
        If given, checks that all entries are in ``[0, n_samples]``.
    check_sorted : bool, default=False
        If ``True``, checks that each row is strictly increasing, i.e.
        ``interval_specs[i, 0] < interval_specs[i, 1] < ...`` for every row.
    min_size : int or None, default=None
        If given, checks that adjacent entries in each row differ by at least
        ``min_size``, i.e. ``interval_specs[i, j+1] - interval_specs[i, j] >= min_size``
        for every row and column pair. Implies strict ordering when ``min_size >= 1``.
    caller_name : str or None, default=None
        Name of the calling function or class. Used in error messages.
    arg_name : str, default="interval_specs"
        Name of the argument being validated. Used in error messages.

    Returns
    -------
    interval_specs : ndarray of shape (n_interval_specs, n_cols)
        Validated array with dtype ``np.intp``.

    Raises
    ------
    ValueError
        If any check fails.
    """
    interval_specs = check_array(interval_specs, ensure_2d=True, dtype=np.intp)

    if interval_specs.shape[1] != n_cols:
        raise ValueError(
            f"`{arg_name}` must have {n_cols} columns, "
            f"got {interval_specs.shape[1]} in {caller_name}."
        )

    if interval_specs.size > 0 and (check_sorted or min_size is not None):
        diffs = np.diff(interval_specs, axis=1)
        if check_sorted and min_size is None and np.any(diffs <= 0):
            raise ValueError(
                f"Each row in `{arg_name}` must be strictly increasing "
                f"(i.e. {arg_name}[i, 0] < {arg_name}[i, 1] < ...) in {caller_name}."
            )
        if min_size is not None and np.any(diffs < min_size):
            raise ValueError(
                f"Adjacent entries in each row of `{arg_name}` must differ by at "
                f"least {min_size} "
                f"(i.e. {arg_name}[i, j+1] - {arg_name}[i, j] >= {min_size}) "
                f"in {caller_name}."
            )

    if n_samples is not None and interval_specs.size > 0:
        out_of_range = interval_specs[
            (interval_specs < 0) | (interval_specs > n_samples)
        ]
        if out_of_range.size > 0:
            raise ValueError(
                f"`{arg_name}` entries must be in [0, {n_samples}], "
                f"got e.g. {out_of_range[0]} in {caller_name}."
            )

    return interval_specs


def check_interval_scorer(
    scorer: "BaseIntervalScorer",
    *,
    ensure_score_type: list | None = None,
    ensure_penalised: bool = False,
    allow_penalised: bool = True,
    caller_name: str | None = None,
    arg_name: str = "",
) -> None:
    """Check if the given scorer is a valid interval scorer.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        The scorer to check.
    ensure_score_type : list of str or None, default=None
        If specified, the scorer's score_type tag must be one of these.
    ensure_penalised : bool, default=False
        If True, raises an error if the scorer is not penalised.
    allow_penalised : bool, default=True
        If False, raises an error if the scorer is penalised.
    caller_name : str or None, default=None
        Name of the caller for error messages.
    arg_name : str, default=""
        Name of the argument for error messages.

    Raises
    ------
    ValueError
        If any of the checks fail.

    """
    score_type = scorer.__sklearn_tags__().interval_scorer_tags.score_type
    if ensure_score_type and score_type not in ensure_score_type:
        _required_tasks = [f'"{task}"' for task in ensure_score_type]
        tasks_str = (
            ", ".join(_required_tasks[:-1]) + " or " + _required_tasks[-1]
            if len(_required_tasks) > 1
            else _required_tasks[0]
        )
        raise ValueError(
            f"{caller_name} requires `{arg_name}` to have score_type {tasks_str}"
            f" ({arg_name}.__sklearn_tags__().interval_scorer_tags.score_type "
            f"in {ensure_score_type}). "
            f'Got {scorer.__class__.__name__}, which has score_type "{score_type}".'
        )
    if (
        ensure_penalised
        and not scorer.__sklearn_tags__().interval_scorer_tags.penalised
    ):
        raise ValueError(
            f"{caller_name} requires `{arg_name}` to be a penalised scorer "
            f"({arg_name}.__sklearn_tags__().interval_scorer_tags.penalised == True). "
            f"Got {scorer.__class__.__name__}, which is not penalised. "
            f"Wrap it with PenalisedScore: "
            f"PenalisedScore({scorer.__class__.__name__}())."
        )
    if not allow_penalised and scorer.__sklearn_tags__().interval_scorer_tags.penalised:
        raise ValueError(
            f"{caller_name} requires `{arg_name}` to be an unpenalised scorer "
            f"({arg_name}.__sklearn_tags__().interval_scorer_tags.penalised == False). "
            f"Got {scorer.__class__.__name__}, which is penalised."
        )


def check_penalty(
    penalty: float | ArrayLike,
    *,
    ensure_non_decreasing: bool = True,
    copy: bool = True,
    caller_name: str | None = None,
    arg_name: str = "penalty",
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
    arg_name : str, default="penalty"
        The name of the argument. Used for error messages.
    """
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
