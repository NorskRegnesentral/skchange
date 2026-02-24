"""Utility functions for the new skchange API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import clone as sklearn_clone
from sklearn.utils._tags import InputTags, Tags, TargetTags

from skchange.new_api.typing import ArrayLike, Segmentation

if TYPE_CHECKING:
    from skchange.new_api.scorers import BaseIntervalScorer


@dataclass(slots=True)
class SkchangeInputTags(InputTags):
    """Extended input tags for skchange estimators.

    Extends sklearn's InputTags with additional skchange-specific input constraints.

    Attributes
    ----------
    multivariate : bool, default=True
        Whether the estimator can handle multivariate data (n_features > 1).
    integer_only : bool, default=False
        Whether the estimator requires integer-valued input data (e.g., for count data).
    """

    multivariate: bool = True
    integer_only: bool = False


@dataclass(slots=True)
class ChangeDetectorTags:
    """Tags specific to change detection estimators.

    Attributes
    ----------
    variable_identification : bool, default=False
        Whether the detector can identify which variables are affected
        at each changepoint.
    """

    variable_identification: bool = False


@dataclass(slots=True)
class IntervalScorerTags:
    """Tags specific to interval scorer estimators.

    Attributes
    ----------
    score_type : str, default="cost"
        Type of score: "cost", "change_score", "saving", "local_saving".
    conditional : bool, default=False
        Whether the scorer uses some input variables as covariates.
        If True, requires at least two input variables.
    aggregated : bool, default=False
        Whether the scorer always returns a single value per cut,
        irrespective of input data shape.
    penalised : bool, default=False
        Whether the score is inherently penalised. If True, score > 0
        indicates change/anomaly. If False, external penalisation needed.
    """

    score_type: str = "cost"
    conditional: bool = False
    aggregated: bool = False
    penalised: bool = False


@dataclass
class SkchangeTags(Tags):
    """Extended tags for skchange estimators.

    Extends sklearn's base Tags with change detection specific tag groups.

    Attributes
    ----------
    input_tags : SkchangeInputTags
        Extended input data tags with skchange-specific constraints.
    change_detector_tags : ChangeDetectorTags
        Change detection specific tags.
    interval_scorer_tags : IntervalScorerTags
        Interval scorer specific tags.
    """

    # Re-declare required parent fields with defaults
    # Must be declared at the top of the class to avoid dataclass ordering issues.
    estimator_type: str | None = None
    target_tags: TargetTags = field(default_factory=lambda: TargetTags(required=False))
    input_tags: SkchangeInputTags = field(default_factory=SkchangeInputTags)

    # New fields.
    # The presence of one of these tag classes indicate the type of estimator.
    change_detector_tags: ChangeDetectorTags | None = None
    interval_scorer_tags: IntervalScorerTags | None = None


def make_segmentation(
    changepoints: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    changed_features: list[np.ndarray] | None = None,
) -> Segmentation:
    """Create a Segmentation dict with clean syntax.

    This helper function mimics dataclass-style construction while returning
    a plain dict. Keeps the sparse representation minimal - labels are only
    included if explicitly provided.

    Required Parameters
    -------------------
    changepoints : np.ndarray
        Changepoint indices, shape (n_changepoints,).

    Optional Parameters
    -------------------
    labels : np.ndarray | None, default=None
        Segment labels, shape (n_changepoints + 1,).
        **If None (default), labels are NOT included in output.**
        Auto-generation happens lazily when converting to dense (e.g., via
        transform() or sparse_to_dense()). Only provide labels explicitly
        for recurring patterns (e.g., [0, 1, 0, 1] for alternating states).
    changed_features : list[np.ndarray] | None, default=None
        Features affected at each changepoint. Only added to output if not None.

    Returns
    -------
    Segmentation
        A typed dict with the specified fields

    Examples
    --------
    >>> # Minimal usage - just changepoints
    >>> result = make_segmentation(
    ...     changepoints=np.array([50, 100]),
    ... )
    >>> result.keys()
    dict_keys(['changepoints'])
    >>> "labels" in result
    False

    >>> # Explicit labels for recurring patterns
    >>> result = make_segmentation(
    ...     changepoints=np.array([50, 100]),
    ...     labels=np.array([0, 1, 0]),  # Return to state 0
    ... )
    >>> result["labels"]
    array([0, 1, 0])
    """
    # Basic validation - fast checks for common errors
    changepoints = np.asarray(changepoints, dtype=int)
    if changepoints.ndim != 1:
        raise ValueError(
            f"changepoints must be 1D array, got shape {changepoints.shape}"
        )
    result: Segmentation = {
        "changepoints": changepoints,
    }

    if labels is not None:
        labels = np.asarray(labels, dtype=int)
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D array, got shape {labels.shape}")
        expected_len = len(changepoints) + 1
        if len(labels) != expected_len:
            raise ValueError(
                f"labels must have length {expected_len} (n_changepoints + 1), "
                f"got {len(labels)}"
            )
        result["labels"] = labels

    if changed_features is not None:
        # Ensure all elements are integer arrays
        validated_features = []
        for i, feats in enumerate(changed_features):
            feats_arr = np.asarray(feats, dtype=int)
            if feats_arr.ndim != 1:
                msg = (
                    f"changed_features[{i}] must be 1D array, "
                    f"got shape {feats_arr.shape}"
                )
                raise ValueError(msg)
            validated_features.append(feats_arr)
        result["changed_features"] = validated_features

    return result


def validate_segmentation(
    result: Segmentation, n_samples: int | None = None, strict: bool = True
) -> None:
    """Validate a Segmentation dict for correctness.

    Performs thorough validation of a Segmentation result, checking that all
    fields are properly formatted and internally consistent.

    Parameters
    ----------
    result : Segmentation
        Segmentation dict to validate.
    n_samples : int | None, optional
        Number of samples in the time series. If provided, validates that
        changepoints are within valid range [0, n_samples).
    strict : bool, default=True
        If True, performs all validations including checking that changepoints
        are sorted and within valid range. If False, only checks types and shapes.

    Raises
    ------
    ValueError
        If any validation check fails.
    TypeError
        If fields have incorrect types.

    Examples
    --------
    >>> result = make_segmentation(
    ...     changepoints=np.array([50, 100]),
    ... )
    >>> validate_segmentation(result, n_samples=200)  # No error - valid

    >>> # Invalid - changepoint out of range
    >>> bad_result = {
    ...     "changepoints": np.array([50, 250]),
    ... }
    >>> validate_segmentation(bad_result, n_samples=200)  # Raises ValueError
    """
    # Check required fields
    if "changepoints" not in result:
        raise ValueError("Segmentation must have 'changepoints' field")

    changepoints = result["changepoints"]

    # Validate changepoints
    if not isinstance(changepoints, np.ndarray):
        raise TypeError(
            f"changepoints must be np.ndarray, got {type(changepoints).__name__}"
        )
    if changepoints.ndim != 1:
        raise ValueError(f"changepoints must be 1D, got shape {changepoints.shape}")
    if not np.issubdtype(changepoints.dtype, np.integer):
        raise TypeError(
            f"changepoints must have integer dtype, got {changepoints.dtype}"
        )

    if strict and len(changepoints) > 0:
        # Check changepoints are sorted
        if not np.all(changepoints[:-1] < changepoints[1:]):
            raise ValueError("changepoints must be strictly increasing")

        # Check changepoints are in valid range if n_samples provided
        if changepoints[0] < 0:
            raise ValueError(f"changepoints must be >= 0, got {changepoints[0]}")
        if n_samples is not None:
            if changepoints[-1] >= n_samples:
                raise ValueError(
                    f"changepoints must be < n_samples ({n_samples}), "
                    f"got {changepoints[-1]}"
                )

    # Validate optional fields if present
    if "labels" in result:
        labels = result["labels"]
        if not isinstance(labels, np.ndarray):
            raise TypeError(f"labels must be np.ndarray, got {type(labels).__name__}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D, got shape {labels.shape}")
        expected_len = len(changepoints) + 1
        if len(labels) != expected_len:
            raise ValueError(
                f"labels must have length {expected_len}, got {len(labels)}"
            )

    if "changed_features" in result:
        changed_features = result["changed_features"]
        if not isinstance(changed_features, list):
            raise TypeError(
                f"changed_features must be list, got {type(changed_features).__name__}"
            )
        if len(changed_features) != len(changepoints):
            raise ValueError(
                f"changed_features must have length {len(changepoints)}, "
                f"got {len(changed_features)}"
            )
        for i, feats in enumerate(changed_features):
            if not isinstance(feats, np.ndarray):
                raise TypeError(
                    f"changed_features[{i}] must be np.ndarray, "
                    f"got {type(feats).__name__}"
                )
            if feats.ndim != 1:
                raise ValueError(
                    f"changed_features[{i}] must be 1D, got shape {feats.shape}"
                )


def sparse_to_dense(result: Segmentation, n_samples: int) -> np.ndarray:
    """Convert sparse segmentation to dense segment labels.

    Parameters
    ----------
    result : Segmentation
        Dict with required field:

        - "changepoints": np.ndarray of changepoint indices

        Optional fields:

        - "labels": np.ndarray of segment labels.
          If not provided, auto-generates [0, 1, 2, ...]

    n_samples : int
        Number of samples in the time series. Required for creating
        the dense array of the correct length.

    Returns
    -------
    np.ndarray
        Dense labels, shape (n_samples,). Each sample assigned its segment label.

    Examples
    --------
    >>> # With explicit labels
    >>> result = {
    ...     "changepoints": np.array([50, 100]),
    ...     "labels": np.array([0, 1, 2]),
    ... }
    >>> dense_labels = sparse_to_dense(result, n_samples=150)
    >>> dense_labels.shape
    (150,)
    >>> np.unique(dense_labels)
    array([0, 1, 2])

    >>> # Without labels (auto-generated)
    >>> result = {
    ...     "changepoints": np.array([50, 100]),
    ... }
    >>> dense_labels = sparse_to_dense(result, n_samples=150)
    >>> np.unique(dense_labels)
    array([0, 1, 2])
    """
    changepoints = np.asarray(result["changepoints"], dtype=int)

    # Get labels, auto-generate if not provided
    if "labels" in result:
        labels = np.asarray(result["labels"], dtype=int)
    else:
        # Auto-generate unique labels for each segment
        n_changepoints = len(changepoints) if changepoints is not None else 0
        labels = np.arange(n_changepoints + 1, dtype=int)

    dense_labels = np.zeros(n_samples, dtype=int)

    if len(changepoints) > 0:
        # Create segment boundaries
        boundaries = np.concatenate([[0], changepoints, [n_samples]])
        for seg_id in range(len(boundaries) - 1):
            start = boundaries[seg_id]
            end = boundaries[seg_id + 1]
            dense_labels[start:end] = labels[seg_id]

    return dense_labels


def dense_to_sparse(labels: np.ndarray) -> Segmentation:
    """Convert dense segment labels to sparse segmentation.

    Parameters
    ----------
    labels : np.ndarray
        Dense segment labels, shape (n_samples,). Each segment should have
        a consistent label.

    Returns
    -------
    Segmentation
        Dict with required field:

        - "changepoints": np.ndarray, extracted from label transitions

        Optional field:

        - "labels": np.ndarray, segment labels

    Examples
    --------
    >>> # Dense labels with 3 segments
    >>> labels = np.array([0,0,0,1,1,1,2,2,2])
    >>> result = dense_to_sparse(labels)
    >>> result["changepoints"]
    array([3, 6])
    >>> result["labels"]
    array([0, 1, 2])

    >>> # Labels with repeated segments (same label returns)
    >>> labels = np.array([0,0,1,1,2,2,0,0])
    >>> result = dense_to_sparse(labels)
    >>> result["changepoints"]
    array([2, 4, 6])
    >>> result["labels"]
    array([0, 1, 2, 0])
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D array. Got shape {labels.shape}")

    # Find changepoints where label changes
    if len(labels) == 0:
        changepoints = np.array([], dtype=int)
        segment_labels = np.array([], dtype=int)
    else:
        # Indices where labels change
        changes = np.where(np.diff(labels) != 0)[0] + 1
        changepoints = changes.astype(int)

        # Extract segment labels
        segment_starts = np.concatenate([[0], changepoints])
        segment_labels = labels[segment_starts].astype(int)

    return make_segmentation(
        changepoints=changepoints,
        labels=segment_labels,
    )


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
    penalty : float | np.ndarray
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
        from skchange.new_api.scorers import ChangeScore

        return ChangeScore(scorer)

    if caller_name is None:
        caller_name = "to_change_score"
    raise ValueError(
        f"`{arg_name}` in {caller_name} must have score_type 'cost' or "
        f"'change_score'. Got score_type '{score_type}'."
    )
