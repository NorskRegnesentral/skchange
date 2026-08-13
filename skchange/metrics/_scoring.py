"""Detector scorers and a metric-to-scorer adapter.

A *scorer* in the sklearn sense is a callable ``(detector, X, y=None) -> float`` that
evaluates a fitted detector on data. This module provides:

.. note::
    Unlike sklearn scorers, scorers here are **not** required to follow the
    higher-is-better convention.
"""

from typing import Any, Callable

from skchange.new_api.types import ArrayLike
from skchange.new_api.utils._param_validation import validate_params


def n_changepoints(detector, X: ArrayLike, y: ArrayLike | None = None) -> float:
    return float(len(detector.predict(X)))


def n_segment_anomalies(detector, X: ArrayLike, y: ArrayLike | None = None) -> float:
    return float(len(detector.predict_segment_anomalies(X)))


def n_segments(detector, X: ArrayLike, y: ArrayLike | None = None) -> float:
    return float(len(detector.predict(X)) + 1)


BUILTIN_SCORERS: dict[str, Callable[[Any, ArrayLike, ArrayLike | None], float]] = {
    "n_changepoints": n_changepoints,
    "n_segment_anomalies": n_segment_anomalies,
    "n_segments": n_segments,
}


@validate_params(
    {"scoring": [str, callable]},
    prefer_skip_nested_validation=True,
)
def resolve_scoring(
    scoring: str | Callable,
) -> Callable[[Any, ArrayLike, ArrayLike | None], float]:
    """Normalise ``scoring`` into a callable ``(detector, X, y) -> float``."""
    if isinstance(scoring, str):
        if scoring not in BUILTIN_SCORERS:
            raise ValueError(
                f"Unknown scoring name {scoring!r}; "
                f"valid names are {sorted(BUILTIN_SCORERS)}."
            )
        return BUILTIN_SCORERS[scoring]
    return scoring


def make_detector_scorer(
    metric: Callable,
    *,
    response_method: str = "predict",
) -> Callable[[Any, ArrayLike, ArrayLike | None], float]:
    """Wrap a metric as a detector scorer.

    Returns a callable ``(detector, X, y) -> float`` that calls
    ``response_method`` on ``detector`` and forwards
    ``(y, y_pred, n_samples=len(X))`` to ``metric``. Mirrors
    :func:`sklearn.metrics.make_scorer`'s ``response_method`` argument.

    Parameters
    ----------
    metric : callable
        A metric from :mod:`skchange.new_api.metrics` with signature
        ``(y_true, y_pred, n_samples, ...)``.
    response_method : str, default="predict"
        Name of the detector method that supplies ``y_pred``. Typical
        values: ``"predict"``, ``"predict_segment_anomalies"``.
    """

    def scorer(detector, X: ArrayLike, y: ArrayLike | None = None) -> float:
        if y is None:
            raise ValueError("`y` is required for metric-based scorers; got None.")
        y_pred = getattr(detector, response_method)(X)
        return float(metric(y, y_pred, n_samples=len(X)))

    return scorer
