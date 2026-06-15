"""Generic FWER calibration for change/anomaly detectors.

A detector's default penalty controls how easily it fires, but BIC-style
defaults are functions of ``(n_samples, n_features)`` only and can be badly
miscalibrated on short series -- producing far more false alarms than intended.

This module calibrates a detector's scalar ``penalty_scale`` so that the
**family-wise error rate** (FWER), the probability of flagging *at least one*
change on change-free data, matches a target ``level``.

For each null sample the module finds the *critical* penalty scale -- the
smallest ``penalty_scale`` that suppresses every detection -- and returns the
``(1 - level)`` quantile of those critical scales.

**Two mechanisms** are used, selected automatically by the detector's
``_calibration_strategy`` class attribute:

- ``"max_score"`` (scan-and-threshold detectors: SBS, MW, CBS, CAPA):
  closed-form ``c_b = max(S) / base``, where ``S`` is the vector of
  unpenalised interval scores. Exact because these detectors threshold each
  score independently. One fit per null sample.

- ``"detection_count"`` (default, and the strategy used by PELT): bisect
  ``penalty_scale`` until the number of detections hits zero. Exact for any
  detector with a single ``penalty_scale`` knob; no structural assumption
  required. About 15-25 fits per null sample.

  PELT uses this strategy because it optimises jointly over all changepoint
  sets: the best single-split score underestimates the true critical penalty,
  so ``"max_score"`` would leave the FWER uncontrolled.

Detectors with no single ``penalty_scale`` knob (e.g. ``CROPS``, which
searches a penalty *range*) raise a clear :class:`ValueError`.
"""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import available_if

from skchange.new_api.tuning._null_models import make_null_draw
from skchange.new_api.tuning._penalty_calibration import unpenalised_scores
from skchange.new_api.utils._param_validation import (
    HasMethods,
    Interval,
    StrOptions,
    validate_params,
)
from skchange.new_api.utils.validation import check_is_fitted, validate_data

# --------------------------------------------------------------------------- #
# Knob discovery (D4)
# --------------------------------------------------------------------------- #


def _discover_knob(detector, X) -> tuple[str, float | None]:
    """Return ``(knob_name, base_penalty)`` for the given detector.

    The ``base_penalty`` is ``penalty_`` at ``penalty_scale=1`` (a positive
    scalar), or ``None`` when the effective penalty is not a simple scalar
    (e.g. CAPA with two bases, or an ESAC-based detector where the penalty
    lives inside the scorer).

    Parameters
    ----------
    detector : estimator
        Any skchange detector (not necessarily fitted).
    X : ndarray of shape (n_samples, n_features)
        Data used to fit a probe copy at ``penalty_scale=1``.

    Returns
    -------
    knob : str
        Parameter name to sweep during calibration, e.g. ``"penalty_scale"``
        or ``"change_score__penalty_scale"``.
    base : float or None
        The unit penalty at ``knob=1``, or ``None`` when not a scalar.

    Raises
    ------
    ValueError
        If the detector has no single ``penalty_scale`` knob (e.g. CROPS).
    """
    params = detector.get_params()

    # Case 1: top-level penalty_scale present.
    if "penalty_scale" in params:
        fitted = clone(detector).set_params(penalty_scale=1.0).fit(X)
        penalty_ = getattr(fitted, "penalty_", _SENTINEL)

        if penalty_ is _SENTINEL:
            # Detector has penalty_scale but no standard penalty_ attr (e.g. CAPA).
            return "penalty_scale", None

        if penalty_ is None:
            # penalty_ is None -> inherently penalised scorer at top level.
            # Fall through to nested search.
            pass
        else:
            base_arr = np.asarray(penalty_, dtype=np.float64).ravel()
            if base_arr.size == 1:
                return "penalty_scale", float(base_arr[0])
            # Array-valued penalty_ -> can only use detection_count.
            return "penalty_scale", None

    # Case 2: search for a nested penalty_scale (e.g. change_score__penalty_scale
    # when the scorer is inherently penalised, like ESACScore).
    for param in sorted(params):
        if param.endswith("__penalty_scale"):
            try:
                clone(detector).set_params(**{param: 1.0}).fit(X)
                return param, None
            except Exception:  # noqa: S112
                continue

    raise ValueError(
        f"{type(detector).__name__!r} has no single `penalty_scale` knob and "
        f"cannot be calibrated. Detectors that search a penalty range "
        f"(e.g. CROPS) are not supported. Use a detector with a scalar "
        f"`penalty_scale` parameter instead."
    )


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


# --------------------------------------------------------------------------- #
# Critical-scale computation
# --------------------------------------------------------------------------- #


def _critical_scale_max_score(detector, X: np.ndarray, knob: str, base: float) -> float:
    """Closed-form critical scale via ``c_b = max(S) / base``.

    Uses the detector's ``_unpenalised_change_scores`` hook to obtain ``S``.
    For scan-and-threshold detectors the hook returns ``unpenalised_scores``;
    for PELT it returns single-split cost reductions.

    Parameters
    ----------
    detector : estimator
        Must be fit-ready (not necessarily fitted).
    X : ndarray of shape (n_samples, n_features)
        Null sample.
    knob : str
        The penalty parameter name (always ``"penalty_scale"`` for this path).
    base : float
        Unit penalty at ``penalty_scale=1``.

    Returns
    -------
    c_b : float
        Critical scale for this null sample.
    """
    fitted = clone(detector).set_params(**{knob: 1.0}).fit(X)
    if hasattr(fitted, "_unpenalised_change_scores"):
        S = np.asarray(fitted._unpenalised_change_scores(X), dtype=np.float64)
    else:
        S = np.asarray(
            unpenalised_scores(detector, X, knob, no_penalty_value=0.0),
            dtype=np.float64,
        )
    max_s = float(S.max()) if S.size else 0.0
    return max(max_s, 0.0) / base


_BISECT_LO = 1e-10  # smallest penalty_scale tested; avoids the scale=0 constraint edge


def _critical_scale_count(
    detector,
    X: np.ndarray,
    knob: str,
    max_scale: float = 1e6,
    rtol: float = 1e-4,
) -> float:
    """Critical scale via bisection on the detected count.

    Finds the smallest ``penalty_scale`` at which the detector reports zero
    detections, using a monotone bisection. Works for any detector with a
    single penalty knob; makes no convexity assumption.

    Parameters
    ----------
    detector : estimator
    X : ndarray of shape (n_samples, n_features)
    knob : str
    max_scale : float
        Upper bracket guard.
    rtol : float
        Relative bisection tolerance.

    Returns
    -------
    c_b : float
    """

    def still_fires(scale: float) -> bool:
        fitted = clone(detector).set_params(**{knob: scale}).fit(X)
        return len(fitted.predict_changepoints(X)) > 0

    # Start from a near-zero scale (not literally 0 to respect parameter constraints).
    if not still_fires(_BISECT_LO):
        return _BISECT_LO

    hi = 1.0
    while still_fires(hi):
        hi *= 2.0
        if hi > max_scale:
            return float(max_scale)

    lo = _BISECT_LO
    while hi - lo > rtol * max(hi, 1.0):
        mid = 0.5 * (lo + hi)
        if still_fires(mid):
            lo = mid
        else:
            hi = mid
    return float(hi)


def _compute_critical_scale(
    detector,
    X: np.ndarray,
    knob: str,
    base: float | None,
    calibration_strategy: str,
    max_scale: float,
    rtol: float,
) -> float:
    """Dispatch to the right critical-scale mechanism.

    Uses ``"max_score"`` (closed form, O(1) fit) when the detector is tagged
    for it AND ``base`` is a known scalar. Falls back to ``"detection_count"``
    bisection otherwise.
    """
    use_max_score = (
        calibration_strategy == "max_score"
        and base is not None
        and knob == "penalty_scale"
    )
    if use_max_score:
        assert base is not None  # guarded by use_max_score
        return _critical_scale_max_score(detector, X, knob, base)
    return _critical_scale_count(detector, X, knob, max_scale, rtol)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


_VALID_STRATEGIES = {"max_score", "detection_count"}


@validate_params(
    {
        "detector": [
            HasMethods(["fit", "set_params", "get_params", "predict_changepoints"])
        ],
        "X": ["array-like"],
        "X_calib": [None, "array-like"],
        "sampler": [str, callable, HasMethods(["sample"])],
        "level": [Interval(Real, 0, 1, closed="neither")],
        "n_simulations": [Interval(Integral, 1, None, closed="left")],
        "calibration_strategy": [StrOptions(_VALID_STRATEGIES), None],
        "max_scale": [Interval(Real, 0, None, closed="neither")],
        "rtol": [Interval(Real, 0, None, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
def calibrate_penalty_scale(
    detector,
    X,
    *,
    X_calib=None,
    sampler="permutation",
    level: float = 0.05,
    n_simulations: int = 999,
    calibration_strategy: str | None = None,
    random_state=None,
    max_scale: float = 1e6,
    rtol: float = 1e-4,
) -> float:
    """Calibrate a detector's ``penalty_scale`` to control the FWER.

    Draws ``n_simulations`` change-free ("null") samples, computes the critical
    penalty scale on each (the smallest ``penalty_scale`` that suppresses all
    detections), and returns the ``(1 - level)`` quantile -- the scale at which
    the probability of at least one false detection is approximately ``level``.

    Two calibration mechanisms are available, selected by the detector's
    ``_calibration_strategy`` class attribute (or overridden by
    ``calibration_strategy``):

    - ``"max_score"`` (scan-and-threshold detectors: SBS, MW, CBS, CAPA):
      closed-form ``c_b = max(S) / base``. Exact because these detectors
      threshold each interval score independently. One fit per null sample.

    - ``"detection_count"`` (PELT and the default fallback): bisects
      ``penalty_scale`` until the detected count hits zero. Exact for any
      detector with a single ``penalty_scale`` knob. About 15-25 fits per
      null sample. PELT uses this strategy because it optimises jointly over
      all changepoint sets; the single-split score underestimates the true
      critical penalty.

    Parameters
    ----------
    detector : estimator
        A detector exposing ``fit``, ``set_params``, ``get_params``,
        ``predict_changepoints``, and a scalar ``penalty_scale``. Not modified.
    X : array-like of shape (n_samples, n_features)
        Data to be analysed for changes. Determines the null sample length
        and the base penalty.
    X_calib : array-like of shape (n_calib, n_features), optional
        Separate change-free data used as the null source for data-based
        samplers. When ``None``, the null is drawn from ``X``. Providing clean
        calibration data avoids conservatism when ``X`` contains a real change.
        Ignored by parametric samplers.
    sampler : str, sampler instance, or callable, default="permutation"
        Null model. ``"permutation"`` resamples rows; ``"gaussian"`` draws
        i.i.d. N(0,1). A callable must have signature
        ``f(n_samples, n_features, rng) -> ndarray``.
    level : float, default=0.05
        Target FWER, in ``(0, 1)``.
    n_simulations : int, default=999
        Number of null samples drawn.
    calibration_strategy : {"max_score", "detection_count"} or None, default=None
        Override the detector's own ``_calibration_strategy`` tag. Pass
        ``"detection_count"`` to force the exact bisection path for PELT when
        using non-convex costs.
    random_state : int, Generator, or None, default=None
        Seed or generator for reproducibility.
    max_scale : float, default=1e6
        Upper bound for the bisection bracket.
    rtol : float, default=1e-4
        Relative tolerance for the bisection.

    Returns
    -------
    penalty_scale : float
        Calibrated value to set on ``detector.penalty_scale`` (or the nested
        scorer's ``penalty_scale`` for inherently-penalised scorers).

    Raises
    ------
    ValueError
        If the detector has no single ``penalty_scale`` knob (e.g. CROPS).

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.detectors import SeededBinarySegmentation
    >>> from skchange.new_api.tuning import calibrate_penalty_scale
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(150, 2))
    >>> scale = calibrate_penalty_scale(
    ...     SeededBinarySegmentation(), X, n_simulations=99, random_state=0
    ... )
    >>> scale > 0
    True
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"`X` must be 2-D, got shape {X.shape}.")
    n_samples, n_features = X.shape

    if X_calib is not None:
        X_calib = np.asarray(X_calib, dtype=np.float64)
        if X_calib.ndim != 2:
            raise ValueError(f"`X_calib` must be 2-D, got shape {X_calib.shape}.")
        if X_calib.shape[1] != n_features:
            raise ValueError(
                f"`X_calib` has {X_calib.shape[1]} features but `X` has "
                f"{n_features}. They must match."
            )

    # Discover the knob and base penalty once (anchored to len(X)).
    knob, base = _discover_knob(detector, X)

    # Resolve the calibration strategy.
    if calibration_strategy is None:
        strategy = getattr(detector, "_calibration_strategy", "detection_count")
    else:
        strategy = calibration_strategy

    draw = make_null_draw(sampler, X, X_calib, n_samples, n_features)
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    critical_scales = np.empty(n_simulations, dtype=np.float64)
    for b in range(n_simulations):
        X_null = draw(rng)
        critical_scales[b] = _compute_critical_scale(
            detector, X_null, knob, base, strategy, max_scale, rtol
        )

    return float(np.quantile(critical_scales, 1.0 - level))


def _detector_has(method: str):
    """available_if check: the wrapped detector exposes ``method``."""

    def check(self) -> bool:
        detector = getattr(self, "detector_", None)
        if detector is None:
            detector = self.detector
        return hasattr(detector, method)

    return check


class CalibratedDetector(BaseEstimator):
    """Wrap a detector and calibrate its ``penalty_scale`` for FWER control.

    A meta-estimator in the spirit of scikit-learn's ``CalibratedClassifierCV``:
    ``fit`` calibrates the wrapped detector's ``penalty_scale`` (via
    :func:`calibrate_penalty_scale`), refits the detector with the calibrated
    scale, and exposes it as ``detector_``. Prediction methods delegate to the
    calibrated detector.

    .. warning::
        When the wrapped detector is ``PELT`` (or another detector tagged
        ``"max_score"``), calibration uses the single-split cost-reduction
        shortcut, which is exact **only for convex costs** (L2, etc.). For
        non-convex costs pass ``calibration_strategy="detection_count"`` to
        force the exact bisection path.

    Parameters
    ----------
    detector : estimator
        Detector to calibrate. Must expose a scalar ``penalty_scale`` parameter
        and ``predict_changepoints``.
    sampler : str, sampler instance, or callable, default="permutation"
        Null model passed to :func:`calibrate_penalty_scale`.
    level : float, default=0.05
        Target FWER.
    n_simulations : int, default=999
        Number of null samples.
    calibration_strategy : {"max_score", "detection_count"} or None, default=None
        Override the detector's calibration strategy. See
        :func:`calibrate_penalty_scale`.
    random_state : int, Generator, or None, default=None
        Seed or generator for reproducibility.

    Attributes
    ----------
    detector_ : estimator
        The wrapped detector, fitted with the calibrated ``penalty_scale``.
    penalty_scale_ : float
        The calibrated penalty scale.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.detectors import SeededBinarySegmentation
    >>> from skchange.new_api.tuning import CalibratedDetector
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(150, 2))
    >>> cal = CalibratedDetector(
    ...     SeededBinarySegmentation(), n_simulations=99, random_state=0
    ... ).fit(X)
    >>> cal.penalty_scale_ > 0
    True
    >>> cps = cal.predict_changepoints(X)
    """

    def __init__(
        self,
        detector,
        *,
        sampler="permutation",
        level: float = 0.05,
        n_simulations: int = 999,
        calibration_strategy: str | None = None,
        random_state=None,
    ):
        self.detector = detector
        self.sampler = sampler
        self.level = level
        self.n_simulations = n_simulations
        self.calibration_strategy = calibration_strategy
        self.random_state = random_state

    def fit(self, X, y=None, X_calib=None) -> "CalibratedDetector":
        """Calibrate and fit the wrapped detector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be analysed for changes.
        y : Ignored
            Present for API consistency (detectors are unsupervised).
        X_calib : array-like of shape (n_calib, n_features), optional
            Separate change-free data for the null sampler. See
            :func:`calibrate_penalty_scale`.

        Returns
        -------
        self : CalibratedDetector
        """
        X = validate_data(self, X, reset=True, ensure_2d=True)

        # Discover the knob to know what parameter to set on the detector.
        knob, _ = _discover_knob(self.detector, X)

        self.penalty_scale_ = calibrate_penalty_scale(
            self.detector,
            X,
            X_calib=X_calib,
            sampler=self.sampler,
            level=self.level,
            n_simulations=self.n_simulations,
            calibration_strategy=self.calibration_strategy,
            random_state=self.random_state,
        )
        self.detector_ = (
            clone(self.detector).set_params(**{knob: self.penalty_scale_}).fit(X)
        )
        return self

    @available_if(_detector_has("predict_changepoints"))
    def predict_changepoints(self, X) -> np.ndarray:
        """Delegate to the calibrated detector."""
        check_is_fitted(self, "detector_")
        return self.detector_.predict_changepoints(X)

    @available_if(_detector_has("predict_segment_anomalies"))
    def predict_segment_anomalies(self, X) -> np.ndarray:
        """Delegate to the calibrated detector."""
        check_is_fitted(self, "detector_")
        return self.detector_.predict_segment_anomalies(X)

    @available_if(_detector_has("predict"))
    def predict(self, X) -> np.ndarray:
        """Delegate to the calibrated detector."""
        check_is_fitted(self, "detector_")
        return self.detector_.predict(X)

    @available_if(_detector_has("predict_scores"))
    def predict_scores(self, X, return_index: bool = False):
        """Delegate to the calibrated detector."""
        check_is_fitted(self, "detector_")
        return self.detector_.predict_scores(X, return_index=return_index)

    def __sklearn_tags__(self):
        """Propagate tags from the wrapped detector."""
        return self.detector.__sklearn_tags__()
