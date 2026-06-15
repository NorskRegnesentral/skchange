"""Generic FWER calibration for change/anomaly detectors.

A detector's default penalty controls how easily it fires, but the BIC-style
defaults are functions of ``(n_samples, n_features)`` only and can be badly
miscalibrated on short series -- producing far more false alarms than intended.

This module calibrates a detector's scalar ``penalty_scale`` so that the
**family-wise error rate** (FWER), the probability of flagging *at least one*
change on change-free data, matches a target ``level``.

The method is fully generic and uses only the public detector API
(:func:`~skchange.new_api.tuning.unpenalised_scores` / ``predict_scores`` /
``set_params``). For each null sample it finds the *critical* penalty scale --
the smallest scale that suppresses every detection -- and returns the
``(1 - level)`` quantile of those critical scales.

For supported detectors (e.g. :class:`SeededBinarySegmentation`,
:class:`MovingWindow`, :class:`CircularBinarySegmentation`) a detection occurs
exactly when a penalised score exceeds zero, so the critical scale has the closed
form ``max(unpenalised_scores) / base_penalty`` when the penalty is scalar. For
array-valued penalties the critical scale is found by a monotone bisection on
``predict_scores``.
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
    validate_params,
)
from skchange.new_api.utils.validation import check_is_fitted, validate_data


def _max_score(detector, X, penalty_param, scale) -> float:
    """Maximum penalised score of ``detector`` on ``X`` at ``penalty_param=scale``.

    Returns ``-inf`` when the detector produces no scores (no candidate
    intervals), which is treated downstream as "no detection".
    """
    fitted = clone(detector).set_params(**{penalty_param: scale}).fit(X)
    scores = np.asarray(fitted.predict_scores(X), dtype=np.float64)
    return float(scores.max()) if scores.size else -np.inf


def _critical_scale_array(
    detector, X, penalty_param, max_scale: float, rtol: float
) -> float:
    """Smallest ``penalty_param`` value with no detection, via bisection.

    Assumes ``max(predict_scores)`` is non-increasing in the penalty scale, which
    holds because a larger penalty is subtracted from every interval's score.
    """
    if _max_score(detector, X, penalty_param, 0.0) <= 0.0:
        return 0.0  # no detection even without any penalty

    # Grow an upper bracket until detection is suppressed.
    hi = 1.0
    while _max_score(detector, X, penalty_param, hi) > 0.0:
        hi *= 2.0
        if hi > max_scale:
            return max_scale

    lo = 0.0
    # Bisection: invariant fires(lo) and not fires(hi).
    while hi - lo > rtol * max(hi, 1.0):
        mid = 0.5 * (lo + hi)
        if _max_score(detector, X, penalty_param, mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return hi


@validate_params(
    {
        "detector": [HasMethods(["fit", "set_params", "get_params", "predict_scores"])],
        "X": ["array-like"],
        "X_calib": [None, "array-like"],
        "sampler": [str, callable, HasMethods(["sample"])],
        "level": [Interval(Real, 0, 1, closed="neither")],
        "n_simulations": [Interval(Integral, 1, None, closed="left")],
        "penalty_param": [str],
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
    penalty_param: str = "penalty_scale",
    random_state=None,
    max_scale: float = 1e6,
    rtol: float = 1e-4,
) -> float:
    """Calibrate a detector's ``penalty_scale`` to control the FWER.

    Draws ``n_simulations`` change-free ("null") samples, computes the critical
    penalty scale on each (the smallest scale that suppresses all detections),
    and returns the ``(1 - level)`` quantile -- the scale at which the
    probability of at least one false detection is approximately ``level``.

    Parameters
    ----------
    detector : estimator
        A detector exposing ``fit``, ``set_params``, ``get_params``,
        ``predict_scores``, and a scalar ``penalty_param``. Not modified.
    X : array-like of shape (n_samples, n_features)
        Data to be analysed for changes. Determines the null sample length
        ``n_samples`` and the base penalty.
    X_calib : array-like of shape (n_calib, n_features), optional
        Separate change-free data used as the null source for data-based
        samplers. When ``None``, the null is drawn from ``X``. Providing clean
        calibration data avoids the over-conservatism that arises when ``X``
        itself contains a real change. Ignored by parametric samplers.
    sampler : str, sampler instance, or callable, default="permutation"
        Null model. ``"permutation"`` resamples rows of the null source;
        ``"gaussian"`` draws i.i.d. ``N(0, 1)``. A callable must have signature
        ``f(n_samples, n_features, rng) -> ndarray``. See
        :mod:`skchange.new_api.tuning._null_models`.

        Note: the calibrated scale is exact only insofar as the sampler
        reproduces the true null distribution. The default ``"permutation"``
        sampler makes no distributional assumptions, but because the calibrated
        scale depends on an upper quantile of a *maximum* statistic -- governed
        by the most extreme resampled values -- it is higher-variance and can be
        mildly anti-conservative (over-firing). When the null family is known, a
        parametric sampler (e.g. ``"gaussian"``) gives tighter control; either
        way, increasing ``n_simulations`` reduces the variance.
    level : float, default=0.05
        Target FWER, in ``(0, 1)``.
    n_simulations : int, default=999
        Number of null samples drawn.
    penalty_param : str, default="penalty_scale"
        Name of the detector's scalar penalty-scale parameter to calibrate.
    random_state : int, Generator, or None, default=None
        Seed or generator for reproducibility.
    max_scale : float, default=1e6
        Upper bound for the array-penalty bisection bracket.
    rtol : float, default=1e-4
        Relative tolerance for the array-penalty bisection.

    Returns
    -------
    penalty_scale : float
        Calibrated value to set on ``detector.penalty_param``.

    Raises
    ------
    NotImplementedError
        If the detector's effective penalty is ``None`` (an inherently
        penalised scorer such as ``ESACScore``); calibrating those is planned
        for a future version.
    ValueError
        If the base penalty is zero, or shapes are inconsistent.

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

    # Resolve the base penalty at penalty_param == 1.0 (the unit baseline), so
    # that setting penalty_param = c yields penalty = c * base. skchange's
    # default penalties depend only on (n_samples, n_features), hence the base is
    # identical across null samples and is computed once here.
    base_detector = clone(detector).set_params(**{penalty_param: 1.0}).fit(X)
    if not hasattr(base_detector, "penalty_"):
        raise NotImplementedError(
            f"{type(detector).__name__} does not expose a fitted `penalty_` "
            f"attribute; generic penalty-scale calibration is not supported."
        )
    base_penalty = base_detector.penalty_
    if base_penalty is None:
        raise NotImplementedError(
            f"{type(detector).__name__} uses an inherently penalised scorer "
            f"(penalty_ is None). Calibrating self-penalised scorers (e.g. "
            f"ESACScore) is not supported yet."
        )

    base_arr = np.asarray(base_penalty, dtype=np.float64).reshape(-1)
    scalar_penalty = base_arr.size == 1
    if scalar_penalty:
        base_scalar = float(base_arr[0])
        if base_scalar <= 0.0:
            raise ValueError(
                f"Base penalty must be positive for scalar calibration; got "
                f"{base_scalar}."
            )

    draw = make_null_draw(sampler, X, X_calib, n_samples, n_features)

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    critical_scales = np.empty(n_simulations, dtype=np.float64)
    for b in range(n_simulations):
        X_null = draw(rng)
        if scalar_penalty:
            scores = unpenalised_scores(
                detector, X_null, penalty_param, no_penalty_value=0.0
            )
            u = float(scores.max()) if scores.size else 0.0
            critical_scales[b] = max(u, 0.0) / base_scalar
        else:
            critical_scales[b] = _critical_scale_array(
                detector, X_null, penalty_param, max_scale, rtol
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

    A meta-estimator in the spirit of scikit-learn's ``GridSearchCV`` /
    ``CalibratedClassifierCV``: ``fit`` calibrates the wrapped detector's
    ``penalty_scale`` (via :func:`calibrate_penalty_scale`), refits the detector
    with the calibrated scale, and exposes it as ``detector_``. Prediction
    methods delegate to the calibrated detector.

    Parameters
    ----------
    detector : estimator
        Detector to calibrate. Must expose a scalar ``penalty_scale`` parameter
        and ``predict_scores``.
    sampler : str, sampler instance, or callable, default="permutation"
        Null model passed to :func:`calibrate_penalty_scale`.
    level : float, default=0.05
        Target FWER.
    n_simulations : int, default=999
        Number of null samples.
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
        random_state=None,
    ):
        self.detector = detector
        self.sampler = sampler
        self.level = level
        self.n_simulations = n_simulations
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
        self.penalty_scale_ = calibrate_penalty_scale(
            self.detector,
            X,
            X_calib=X_calib,
            sampler=self.sampler,
            level=self.level,
            n_simulations=self.n_simulations,
            random_state=self.random_state,
        )
        self.detector_ = (
            clone(self.detector).set_params(penalty_scale=self.penalty_scale_).fit(X)
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
