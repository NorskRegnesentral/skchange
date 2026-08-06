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

**Three mechanisms** are used, selected automatically by the detector's
``change_detector_tags.calibration_strategy`` tag:

- ``"max_score"`` (scan-and-threshold detectors: SBS, MW, CBS, CAPA):
  closed-form ``c_b = max(S) / base``, where ``S`` is the vector of
  unpenalised interval scores. Exact because these detectors threshold each
  score independently. One fit per null sample.

- ``"path_search"`` (PELT): exact secant search on the convex hull of
  cost-vs-number-of-changepoints. Finds ``β* = max_k G_k/k`` (largest
  average cost reduction per changepoint) in ~3-6 PELT fits instead of
  ~15-25. A small upward nudge handles the transition tie. PELT uses this
  strategy because ``"max_score"`` (single-split ``G_1``) underestimates the
  true critical penalty: the map ``k ↦ G_k`` is generally not concave.

- ``"detection_count"`` (default fallback): bisect ``penalty_scale`` until
  the number of detections hits zero. Exact for any detector with a single
  ``penalty_scale`` knob; no structural assumption required. About 15-25 fits
  per null sample.

Detectors with no single ``penalty_scale`` knob (e.g. ``CROPS``, which
searches a penalty *range*) raise a clear :class:`ValueError`.
"""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils import get_tags
from sklearn.utils.metaestimators import available_if
from sklearn.utils.parallel import Parallel, delayed

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
_BISECT_GRID = 32  # geometric grid points used to locate a firing scale when the
#                    detector is silent at near-zero penalty (non-monotone count)
_PATH_SEARCH_NUDGE = 1e-3  # relative upward nudge to break the β* transition tie


def _critical_scale_path_search(
    detector,
    X: np.ndarray,
    knob: str,
    base: float,
    max_iter: int = 50,
) -> float:
    """Critical scale for PELT via exact convex-hull secant search.

    Finds ``β* = max_{k≥1} G_k/k`` where ``G_k`` is the best total cost
    reduction with ``k`` changepoints. Each iteration jumps to the exact hull
    vertex for the current penalty, converging in ~3-6 PELT fits.

    A small upward nudge (``beta* x 1e-3``) breaks the transition tie so that
    PELT at the returned scale reports zero changepoints.
    """
    n = X.shape[0]

    # Fit at a near-zero scale: (1) gets cost_ for C_0; (2) starts iterations.
    fitted = clone(detector).set_params(**{knob: _BISECT_LO}).fit(X)

    # C_0: full-series unpenalized cost (independent of penalty scale).
    cache = fitted.cost_.precompute(X)
    C_0 = float(np.sum(fitted.cost_.evaluate(cache, np.array([[0, n]]))))

    result = fitted.predict_all(X)
    k = len(result["changepoints"])

    if k == 0:
        return float(_BISECT_LO)

    beta = _BISECT_LO * base
    beta_star = 0.0

    for _ in range(max_iter):
        penalized_opt = result["cumulative_optimal_costs"][-1]
        c_k = penalized_opt - k * fitted.penalty_  # unpenalized segmentation cost
        G_k = C_0 - c_k
        if G_k <= 0 or k <= 0:
            break

        beta_new = G_k / k
        if beta_new <= beta:
            # Floating-point convergence at the hull vertex.
            beta_star = beta_new
            break

        beta_star = beta_new
        beta = beta_new
        fitted = clone(detector).set_params(**{knob: beta / base}).fit(X)
        result = fitted.predict_all(X)
        k = len(result["changepoints"])

        if k == 0:
            break

    nudge = _PATH_SEARCH_NUDGE * (1.0 + beta_star)
    return (beta_star + nudge) / base


def _critical_scale_count(
    detector,
    X: np.ndarray,
    knob: str,
    max_scale: float = 1e6,
    rtol: float = 1e-4,
) -> float:
    """Critical scale via a search on the detected count.

    Returns the *upper edge* of the firing region: the smallest
    ``penalty_scale`` such that the detector (and every larger scale) reports
    zero detections. Works for any detector with a single penalty knob; makes
    no convexity assumption.

    The detected count is **not** assumed monotone in ``penalty_scale``. Most
    detectors fire at near-zero penalty and stop firing once the penalty is
    large enough, so a plain bisection from a tiny scale suffices. But some
    detectors are *silent* at near-zero penalty -- CAPA, for example, absorbs
    the whole series into a single anomalous segment when the penalty is ~0, so
    it reports no changepoints there, fires for moderate penalties, then falls
    silent again. For those, the firing region is an interior interval, so we
    locate a firing scale on a geometric grid before bisecting for the upper
    edge.

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

    # Upper bracket: smallest doubling scale (from 1) at which the detector is
    # silent. Valid regardless of the low-penalty behaviour.
    hi = 1.0
    while still_fires(hi):
        hi *= 2.0
        if hi > max_scale:
            return float(max_scale)

    # Lower bracket: a scale that fires.
    if still_fires(_BISECT_LO):
        # Monotone fast path: fires at ~0, so the firing region reaches the
        # bottom and a single down-crossing lies in (_BISECT_LO, hi).
        lo = _BISECT_LO
    else:
        # Silent at ~0 (non-monotone): scan a geometric grid downward from hi
        # and take the largest firing scale as the lower bracket.
        lo = None
        for scale in np.geomspace(_BISECT_LO, hi, _BISECT_GRID)[::-1]:
            if scale < hi and still_fires(float(scale)):
                lo = float(scale)
                break
        if lo is None:
            # Never fires anywhere -> no penalty needed to suppress it.
            return _BISECT_LO

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

    - ``"max_score"``: closed-form ``c_b = max(S)/base`` (one fit).
    - ``"path_search"``: exact convex-hull secant search (PELT only, ~3-6 fits).
    - ``"detection_count"``: bisection fallback (~15-25 fits, any detector).
    """
    if (
        calibration_strategy == "max_score"
        and base is not None
        and knob == "penalty_scale"
    ):
        assert base is not None  # narrowed by condition
        return _critical_scale_max_score(detector, X, knob, base)
    if calibration_strategy == "path_search" and base is not None:
        assert base is not None  # narrowed by condition
        return _critical_scale_path_search(detector, X, knob, base)
    return _critical_scale_count(detector, X, knob, max_scale, rtol)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


_VALID_STRATEGIES = {"max_score", "detection_count", "path_search"}


def _run_single_sim(seed, draw_fn, detector, knob, base, strategy, max_scale, rtol):
    """Execute one null-sample critical-scale simulation.

    Called in parallel by ``calibrate_penalty_scale``. Each invocation draws
    its own null sample from an independent generator seeded by ``seed``
    (a ``numpy.random.SeedSequence`` child), so there is no shared mutable
    RNG state between parallel tasks.
    """
    rng = np.random.default_rng(seed)
    X_null = draw_fn(rng)
    return _compute_critical_scale(
        detector, X_null, knob, base, strategy, max_scale, rtol
    )


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
    n_jobs=None,
    max_scale: float = 1e6,
    rtol: float = 1e-4,
) -> float:
    """Calibrate a detector's ``penalty_scale`` to control the FWER.

    Draws ``n_simulations`` change-free ("null") samples, computes the critical
    penalty scale on each (the smallest ``penalty_scale`` that suppresses all
    detections), and returns the ``(1 - level)`` quantile -- the scale at which
    the probability of at least one false detection is approximately ``level``.

    Three calibration mechanisms are available, selected by the detector's
    ``change_detector_tags.calibration_strategy`` tag (or overridden by
    ``calibration_strategy``):

    - ``"max_score"`` (scan-and-threshold detectors: SBS, MW, CBS, CAPA):
      closed-form ``c_b = max(S) / base``. Exact because these detectors
      threshold each interval score independently. One fit per null sample.

    - ``"path_search"`` (PELT): exact secant search on the convex hull of
      cost-vs-number-of-changepoints. Finds ``β* = max_k G_k/k`` in ~3-6
      PELT fits instead of ~15-25. A small nudge breaks the transition tie.

    - ``"detection_count"`` (default fallback): bisects ``penalty_scale``
      until the detected count hits zero. Exact for any detector with a
      single ``penalty_scale`` knob. About 15-25 fits per null sample.

    The Monte-Carlo loop runs in parallel using ``sklearn.utils.parallel``
    (wraps joblib; no new dependency). Each simulation draws from its own
    independent generator spawned from a parent ``SeedSequence``, so there is
    no shared mutable RNG state between workers. Results are reproducible for
    a fixed ``random_state`` regardless of ``n_jobs``.

    .. note::
        The RNG stream differs from releases before ``n_jobs`` was added. A
        given ``random_state`` integer produces a different calibrated scale
        than before, though all statistical guarantees are unchanged.

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
    calibration_strategy : {"max_score", "detection_count", "path_search"} or None
        Override the detector's own ``calibration_strategy`` tag. Pass
        ``"detection_count"`` to force the bisection path for PELT when the
        cost is non-convex.
    random_state : int, Generator, or None, default=None
        Seed or generator. Used to derive a parent ``SeedSequence`` from
        which one independent child generator is spawned per simulation.
        Results are reproducible for a fixed value and invariant to ``n_jobs``.
    n_jobs : int or None, default=None
        Number of parallel workers. ``None`` uses scikit-learn's default
        (serial unless a ``joblib.parallel_backend`` context is active);
        ``-1`` uses all available cores. See :class:`joblib.Parallel`.
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
        strategy = get_tags(detector).change_detector_tags.calibration_strategy
    else:
        strategy = calibration_strategy

    draw = make_null_draw(sampler, X, X_calib, n_samples, n_features)

    # Design-B thread-safe RNG: spawn one independent child seed per simulation.
    # The result is reproducible for a fixed random_state and invariant to n_jobs.
    if isinstance(random_state, np.random.Generator):
        entropy = int(random_state.integers(2**62))
    elif random_state is None:
        entropy = None
    else:
        entropy = int(random_state)
    child_seeds = np.random.SeedSequence(entropy).spawn(n_simulations)

    critical_scales = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_sim)(
            seed, draw, detector, knob, base, strategy, max_scale, rtol
        )
        for seed in child_seeds
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

    PELT uses the exact ``"path_search"`` strategy (``β* = max_k G_k/k``)
    by default. Pass ``calibration_strategy="detection_count"`` to force the
    bisection fallback for non-standard costs.

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
    calibration_strategy : {"max_score", "detection_count", "path_search"} or None
        Override the detector's calibration strategy. See
        :func:`calibrate_penalty_scale`.
    random_state : int, Generator, or None, default=None
        Seed or generator. Passed to :func:`calibrate_penalty_scale`; results
        are reproducible and invariant to ``n_jobs``.
    n_jobs : int or None, default=None
        Number of parallel workers for the Monte-Carlo loop. ``None`` uses
        scikit-learn's default (serial unless a backend context is active).

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
        n_jobs=None,
    ):
        self.detector = detector
        self.sampler = sampler
        self.level = level
        self.n_simulations = n_simulations
        self.calibration_strategy = calibration_strategy
        self.random_state = random_state
        self.n_jobs = n_jobs

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
            n_jobs=self.n_jobs,
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
