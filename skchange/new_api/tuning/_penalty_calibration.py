"""Generic penalty calibration helpers for detectors.

The :func:`unpenalised_scores` helper produces a detector's scoring objective
evaluated with one or more penalty parameters set to their no-penalty value
(zero for additive penalties, one for multiplicative scale parameters, etc.),
by cloning the detector, setting the named parameter(s), refitting, and
calling ``predict_scores``. The returned scores can then be used by external
calibration logic (threshold sweeps, BIC-style criteria, bootstrap thresholds,
etc.) to choose a final penalty value.

Notes
-----
- Penalty parameter names follow sklearn's nested-parameter syntax. Pass
  ``"penalty"`` for a top-level penalty, ``"change_score__penalty"`` for a
  penalty on a nested change score, etc.
- For multi-knob calibration (e.g. ``ESACScore`` with both
  ``penalty_scale_dense`` and ``penalty_scale_sparse``) pass a mapping of
  parameter names to their no-penalty values.
"""

from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

import numpy as np
from sklearn.base import clone

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

    Clones ``detector``, sets the penalty parameter(s) to their no-penalty
    value, refits on ``X``, and returns the output of ``predict_scores`` on
    the same ``X``. Useful as a generic calibration primitive: the returned
    scores can be used to choose a final penalty value via threshold sweeps,
    BIC-style criteria, or bootstrap thresholds.

    The original ``detector`` is not modified.

    Parameters
    ----------
    detector : estimator
        A detector exposing ``fit``, ``predict_scores``, and sklearn's
        ``clone`` / ``set_params`` API. Need not be fitted.
    X : array-like of shape (n_samples, n_features)
        Data to fit and score on.
    penalty_param : str, iterable of str, or mapping, default="penalty_scale"
        Name(s) of the penalty parameter(s) to disable. Nested parameters
        use sklearn's ``"<step>__<param>"`` syntax. Three forms are
        supported:

        - ``str``: set a single parameter to ``no_penalty_value``.
        - iterable of ``str``: set each parameter to the same
          ``no_penalty_value``.
        - ``Mapping[str, Any]``: set each parameter to its mapped value;
          ``no_penalty_value`` is ignored.
    no_penalty_value : float or np.ndarray, default=0.0
        Value that disables the penalty when ``penalty_param`` is a string
        or an iterable of strings. The default ``0.0`` is appropriate for
        additive penalties; pass ``1.0`` for multiplicative scale
        parameters, an array for an array-valued penalty, or a mapping for
        parameter-specific values.
    return_index : bool, default=False
        Forwarded to the detector's ``predict_scores``. If ``True``, the
        returned object is whatever ``predict_scores`` returns in that mode.

    Returns
    -------
    Same as ``detector.predict_scores(X, return_index=return_index)``.

    Examples
    --------
    Calibrate a scalar additive penalty by taking the worst-case unpenalised
    score:

    >>> import numpy as np
    >>> from skchange.new_api.detectors import SeededBinarySegmentation
    >>> from skchange.new_api.tuning import unpenalised_scores
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> detector = SeededBinarySegmentation()
    >>> scores = unpenalised_scores(detector, X)
    >>> calibrated_penalty = float(scores.max())

    Disable multiple multiplicative scale parameters (e.g. ESAC's two
    scales) by setting them to one:

    >>> from skchange.new_api.interval_scorers import ESACScore
    >>> detector = SeededBinarySegmentation(change_score=ESACScore())
    >>> scores = unpenalised_scores(
    ...     detector,
    ...     X,
    ...     penalty_param=[
    ...         "change_score__penalty_scale_dense",
    ...         "change_score__penalty_scale_sparse",
    ...     ],
    ...     no_penalty_value=1.0,
    ... )

    Disable multiple parameters with explicit per-parameter values:

    >>> scores = unpenalised_scores(
    ...     detector,
    ...     X,
    ...     penalty_param={
    ...         "change_score__penalty_scale_dense": 0.0,
    ...         "change_score__penalty_scale_sparse": 0.0,
    ...     },
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
