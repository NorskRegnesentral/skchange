"""CalibratedDetector meta-estimator."""

import re

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.calibration._calibrate import calibrate_penalty_scale
from skchange.new_api.calibration._null_models import (
    BaseDataSampler,
    BaseParametricSampler,
    _resolve_sampler,
)
from skchange.new_api.interval_scorers._penalised_score import PenalisedScore


def _discover_penalty_scale_params(detector) -> list:
    """Return names of top-level params equal to 'penalty_scale' or ending with it.

    If the detector exposes a ``_active_penalty_scale_params()`` method, that
    takes priority so that detectors can exclude params that are disabled at
    runtime (e.g. ``CAPA.point_penalty_scale`` when
    ``include_point_anomalies=False``).
    """
    if hasattr(detector, "_active_penalty_scale_params"):
        return list(detector._active_penalty_scale_params())
    params = detector.get_params(deep=False)
    return [
        name
        for name in params
        if name == "penalty_scale" or name.endswith("_penalty_scale")
    ]


def _get_inner_scorer_for_penalty(temp_det, penalty_param: str):
    """Extract the unpenalised inner scorer for a given penalty_scale param.

    ``temp_det`` must already be fitted (with all penalty scales set to 1.0).

    The naming convention is:

    - ``penalty_scale`` (no prefix) → check fitted attributes ``change_score_``,
      ``transient_score_``, ``scorer_`` (in that order).
    - ``{prefix}_penalty_scale`` → check ``{prefix}_saving_``, ``{prefix}_score_``,
      ``{prefix}_scorer_``.

    Returns the unpenalised (unfitted) inner scorer (``PenalisedScore.scorer``).
    """
    prefix = re.sub(r"_?penalty_scale$", "", penalty_param)

    if prefix:
        candidates = [f"{prefix}_saving_", f"{prefix}_score_", f"{prefix}_scorer_"]
    else:
        candidates = ["change_score_", "transient_score_", "scorer_"]

    for attr_name in candidates:
        fitted_attr = getattr(temp_det, attr_name, None)
        if fitted_attr is not None and isinstance(fitted_attr, PenalisedScore):
            return fitted_attr.scorer  # unfitted per sklearn param convention

    # Last resort: if there is exactly one PenalisedScore fitted attribute, use it.
    penalised_attrs = [
        v for k, v in vars(temp_det).items() if isinstance(v, PenalisedScore)
    ]
    if len(penalised_attrs) == 1:
        return penalised_attrs[0].scorer

    raise NotImplementedError(
        f"Cannot automatically determine the scorer for '{penalty_param}' "
        f"in {type(temp_det).__name__}. No fitted attribute matching "
        f"{candidates} was found. CalibratedDetector requires that each "
        f"'*_penalty_scale' param corresponds to a fitted PenalisedScore "
        f"attribute following the naming convention."
    )


class CalibratedDetector(BaseEstimator):
    """Meta-estimator that calibrates a detector's penalty scales for false alarm control.

    Discovers parameters named ``penalty_scale`` or ending in ``_penalty_scale``
    via ``detector.get_params(deep=False)``.  Each such parameter is calibrated
    independently via Monte Carlo simulation under the null hypothesis of no change,
    with Bonferroni correction applied when more than one penalty scale is found.

    **Auto-discovery convention**: ``CalibratedDetector`` looks for parameters whose
    name is exactly ``"penalty_scale"`` or ends with ``"_penalty_scale"``.  For each
    such parameter, the corresponding inner scorer is resolved by a naming convention
    on the detector's fitted attributes:

    - ``penalty_scale`` → ``change_score_``, ``transient_score_``, or ``scorer_``
    - ``{prefix}_penalty_scale`` → ``{prefix}_saving_``, ``{prefix}_score_``, or
      ``{prefix}_scorer_``

    Parameters
    ----------
    detector : BaseChangeDetector
        The detector to calibrate. Must expose at least one ``*_penalty_scale``
        parameter.
    resampling : str, BaseDataSampler, or BaseParametricSampler, default="permutation"
        Null sampler used for Monte Carlo simulation. String aliases:
        ``"permutation"`` (:class:`PermutationSampler`),
        ``"block_bootstrap"`` (:class:`BlockBootstrapSampler`),
        ``"gaussian"`` (:class:`GaussianMCSampler`).
        Data-based samplers are fitted on ``X_calib`` when provided, otherwise on
        ``X``. Parametric samplers are never fitted.
    level : float, default=0.05
        Target family-wise error rate. When ``k > 1`` penalty scales are
        discovered, each is calibrated at ``level / k`` (Bonferroni).
    n_simulations : int, default=999
        Number of Monte Carlo simulations per penalty scale.
    random_state : int, Generator, or None, default=None
        Controls reproducibility.
    """

    def __init__(
        self,
        detector,
        resampling: "str | BaseDataSampler | BaseParametricSampler" = "permutation",
        level: float = 0.05,
        n_simulations: int = 999,
        random_state=None,
    ):
        self.detector = detector
        self.resampling = resampling
        self.level = level
        self.n_simulations = n_simulations
        self.random_state = random_state

    def fit(
        self, X: np.ndarray, y=None, X_calib: np.ndarray | None = None, X_train: np.ndarray | None = None
    ) -> "CalibratedDetector":
        """Fit the detector with calibrated penalty scales.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data used to determine the target sample size for calibration.
            Null samples are drawn at ``len(X)`` rows; interval specs are
            generated for ``len(X)``.
        y : None
            Ignored. Present for sklearn API compatibility.
        X_calib : np.ndarray of shape (n_calib, n_features) or None, default=None
            Optional separate null (change-free) dataset. Used to fit data-based
            samplers. When ``None``, data-based samplers are fitted on ``X_train``
            when provided, otherwise on ``X``.
        X_train : np.ndarray of shape (n_train, n_features) or None, default=None
            Optional larger training dataset. When provided, the scorer is fitted
            on ``X_train`` (for better parameter estimates) and the final detector
            is also fitted on ``X_train``. ``len(X)`` still determines the null
            sample length and interval specs.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"`X` must be 2-D, got shape {X.shape}.")

        if X_calib is not None:
            X_calib = np.asarray(X_calib, dtype=np.float64)
            if X_calib.ndim != 2:
                raise ValueError(f"`X_calib` must be 2-D, got shape {X_calib.shape}.")
            if X_calib.shape[1] != X.shape[1]:
                raise ValueError(
                    f"`X_calib` has {X_calib.shape[1]} features but `X` has "
                    f"{X.shape[1]}. They must match."
                )

        if X_train is not None:
            X_train = np.asarray(X_train, dtype=np.float64)
            if X_train.ndim != 2:
                raise ValueError(f"`X_train` must be 2-D, got shape {X_train.shape}.")
            if X_train.shape[1] != X.shape[1]:
                raise ValueError(
                    f"`X_train` has {X_train.shape[1]} features but `X` has "
                    f"{X.shape[1]}. They must match."
                )

        # Data used to fit the scorer and the final detector.
        X_fit = X_train if X_train is not None else X

        # Resolve sampler.
        sampler = _resolve_sampler(self.resampling)

        # Discover penalty_scale parameters.
        penalty_params = _discover_penalty_scale_params(self.detector)
        if not penalty_params:
            raise NotImplementedError(
                f"{type(self.detector).__name__} has no 'penalty_scale' or "
                "'*_penalty_scale' parameters. CalibratedDetector requires at least "
                "one."
            )

        # Bonferroni correction across k penalty scales.
        k = len(penalty_params)
        adjusted_level = self.level / k

        # Fit a temporary detector with all penalty_scales=1.0 to extract scorers.
        temp_det = (
            clone(self.detector)
            .set_params(**{p: 1.0 for p in penalty_params})
            .fit(X_fit, y)
        )

        # Shared RNG for reproducibility across penalty params.
        rng = np.random.default_rng(self.random_state)

        # Calibrate each penalty scale independently.
        penalty_scales: dict = {}
        for penalty_param in penalty_params:
            inner_scorer = _get_inner_scorer_for_penalty(temp_det, penalty_param)
            c_star = calibrate_penalty_scale(
                scorer=inner_scorer,
                X=X,
                sampler=sampler,
                interval_specs=None,
                detector=temp_det,
                level=adjusted_level,
                n_simulations=self.n_simulations,
                random_state=rng,
                X_calib=X_calib,
                X_train=X_train,
            )
            penalty_scales[penalty_param] = c_star

        # Fit final detector with calibrated penalty scales on X_fit (X_train if provided, else X).
        self.detector_ = clone(self.detector).set_params(**penalty_scales).fit(X_fit, y)
        self.penalty_scales_ = penalty_scales
        self.n_simulations_done_ = self.n_simulations
        return self

    def predict_changepoints(self, X: np.ndarray) -> np.ndarray:
        """Detect changepoints using the calibrated detector."""
        check_is_fitted(self)
        return self.detector_.predict_changepoints(X)

    def predict(self, X: np.ndarray):
        """Predict using the calibrated detector."""
        check_is_fitted(self)
        return self.detector_.predict(X)
