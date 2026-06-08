"""Generic penalty calibration helpers for detectors."""

from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any, Callable

import numpy as np
from sklearn.base import clone

from skchange.new_api.metrics._scoring import resolve_scoring
from skchange.new_api.types import ArrayLike
from skchange.new_api.utils._param_validation import (
    HasMethods,
    Interval,
    validate_params,
)


@validate_params(
    {
        "detector": [HasMethods(["fit", "set_params", "predict_scores"])],
        "X": ["array-like"],
        "penalty_param": [str, Iterable, Mapping],
        "no_penalty_value": [
            Interval(Real, None, None, closed="neither"),
            "array-like",
        ],
        "return_index": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def unpenalised_scores(
    detector,
    X: ArrayLike,
    penalty_param: str | Iterable[str] | Mapping[str, Any] = "penalty_scale",
    *,
    no_penalty_value: float | np.ndarray = 0.0,
    return_index: bool = False,
):
    """Compute a detector's scoring objective without penalisation.

    Clones ``detector``, sets the named penalty parameter(s) to a
    no-penalty value, refits on ``X``, and returns the output of
    ``predict_scores``. The original ``detector`` is not modified.

    Parameters
    ----------
    detector : estimator
        Exposes ``fit``, ``predict_scores``, and sklearn's ``clone`` /
        ``set_params``. Need not be fitted.
    X : array-like of shape (n_samples, n_features)
        Data to fit and score on.
    penalty_param : str, iterable of str, or mapping, default="penalty_scale"
        Penalty parameter(s) to disable. Nested parameters use sklearn's
        ``"<step>__<param>"`` syntax.

        - ``str`` / iterable of ``str``: each parameter is set to
          ``no_penalty_value``.
        - ``Mapping[str, Any]``: each parameter is set to its mapped
          value; ``no_penalty_value`` is ignored.
    no_penalty_value : float or np.ndarray, default=0.0
        No-penalty value used when ``penalty_param`` is a string or
        iterable. Use ``0.0`` for additive penalties, ``1.0`` for
        multiplicative scales, or an array for array-valued penalties.
    return_index : bool, default=False
        Forwarded to ``predict_scores``.

    Returns
    -------
    Same as ``detector.predict_scores(X, return_index=return_index)``.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.detectors import SeededBinarySegmentation
    >>> from skchange.new_api.tuning import unpenalised_scores
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> scores = unpenalised_scores(SeededBinarySegmentation(), X)
    >>> calibrated_penalty = float(scores.max())

    Disable two multiplicative scales (e.g. ESAC's):

    >>> from skchange.new_api.interval_scorers import ESACScore
    >>> detector = SeededBinarySegmentation(change_score=ESACScore())
    >>> scores = unpenalised_scores(
    ...     detector, X,
    ...     penalty_param=[
    ...         "change_score__penalty_scale_dense",
    ...         "change_score__penalty_scale_sparse",
    ...     ],
    ...     no_penalty_value=1.0,
    ... )
    """
    if isinstance(penalty_param, str):
        params = {penalty_param: no_penalty_value}
    elif isinstance(penalty_param, Mapping):
        params = dict(penalty_param)
    else:
        params = dict.fromkeys(penalty_param, no_penalty_value)

    cal = clone(detector)
    cal.set_params(**params)
    cal.fit(X)
    return cal.predict_scores(X, return_index=return_index)


@validate_params(
    {
        "detector": [HasMethods(["fit", "set_params"])],
        "X": ["array-like"],
        "y": [None, "array-like"],
        "param_name": [str],
        "param_range": ["array-like"],
        "scoring": [str, callable],
    },
    prefer_skip_nested_validation=True,
)
def penalty_curve(
    detector,
    X: ArrayLike,
    y: ArrayLike | None = None,
    *,
    param_name: str,
    param_range: ArrayLike,
    scoring: str | Callable = "n_changepoints",
) -> np.ndarray:
    """Sweep a penalty parameter and return a score curve.

    For each value in ``param_range``, clones ``detector``, sets
    ``param_name`` to that value, refits on ``X``, and evaluates
    ``scoring`` on the fitted detector. The original ``detector`` is not
    modified. Modeled on :func:`sklearn.model_selection.validation_curve`;
    selecting a final value from the curve is left to the caller.

    Parameters
    ----------
    detector : estimator
        Exposes ``fit``, ``set_params``, sklearn's ``clone``, and any
        prediction methods used by ``scoring``.
    X : array-like of shape (n_samples, n_features)
        Data to fit and score on.
    y : array-like, optional, default=None
        Reference passed as the third argument to ``scoring``. Not
        passed to ``detector.fit`` — skchange detectors are
        unsupervised, so ``y`` is purely a scoring reference.
    param_name : str
        Detector parameter to vary. Any name accepted by ``set_params``
        is allowed, including nested ``"<step>__<param>"`` syntax.
    param_range : array-like of shape (n_candidates,)
        Candidate values. Order is preserved in the output.
    scoring : str or callable, default="n_changepoints"
        Either a built-in scorer name (``"n_changepoints"``,
        ``"n_segment_anomalies"``, ``"n_segments"``) or a callable with
        signature ``(detector, X, y=None) -> float``. Built-in scorers
        ignore ``y``.

    Returns
    -------
    curve : np.ndarray of shape (n_candidates,)
        Scorer output per ``param_range`` value, aligned 1:1 with the
        input.

    Raises
    ------
    ValueError
        ``param_range`` is empty or non-real, or ``scoring`` is an
        unknown built-in name.
    TypeError
        ``scoring`` is neither a string nor a callable.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.detectors import SeededBinarySegmentation
    >>> from skchange.new_api.tuning import penalty_curve
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 1))
    >>> param_range = np.array([1.0, 2.0, 3.0, 5.0])
    >>> counts = penalty_curve(
    ...     SeededBinarySegmentation(), X,
    ...     param_name="penalty", param_range=param_range,
    ... )
    >>> selected = param_range[counts <= 1].min()

    Supervised scoring against reference changepoints, using
    :func:`~skchange.new_api.metrics.make_detector_scorer` to wrap a metric::

        from skchange.new_api.metrics import (
            changepoint_f1_score,
            make_detector_scorer,
        )

        f1 = penalty_curve(
            detector, X, y_true_cps,
            param_name="penalty", param_range=param_range,
            scoring=make_detector_scorer(changepoint_f1_score),
        )
    """
    scorer = resolve_scoring(scoring)

    candidates = np.asarray(param_range)
    if candidates.ndim != 1:
        raise ValueError(
            "`param_range` must be 1-dimensional; "
            f"got array of shape {candidates.shape}."
        )
    if candidates.size == 0:
        raise ValueError("`param_range` must contain at least one value.")
    if not all(isinstance(v, Real) and not isinstance(v, bool) for v in candidates):
        raise ValueError("`param_range` must contain real-valued numbers.")

    curve = np.empty(candidates.shape[0], dtype=float)
    for i, value in enumerate(candidates):
        cal = clone(detector)
        cal.set_params(**{param_name: value})
        cal.fit(X)
        curve[i] = float(scorer(cal, X, y))

    return curve
