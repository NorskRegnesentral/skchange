"""CalibratedDetector meta-estimator."""

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.calibration._calibrate import calibrate_penalty
from skchange.new_api.calibration._null_models import (
    BaseNullModel,
    PermutationNullModel,
)
from skchange.new_api.interval_scorers._penalised_score import PenalisedScore


def _find_penalised_score_params(detector) -> dict:
    """Return a mapping {param_name: PenalisedScore_instance} for top-level params."""
    params = detector.get_params(deep=False)
    return {
        name: val for name, val in params.items() if isinstance(val, PenalisedScore)
    }


class CalibratedDetector(BaseEstimator):
    """Meta-estimator that calibrates a detector's penalties for false alarm control.

    Parameters
    ----------
    detector : BaseChangeDetector
        The detector to calibrate. Must have at least one ``PenalisedScore`` param.
    null_model : BaseNullModel or None, default=None
        Null model used for simulation. Defaults to ``PermutationNullModel()``.
    level : float, default=0.05
        Target family-wise error rate. Multiple penalties are calibrated via
        Bonferroni (each to ``level / k`` where k is the number of penalties).
    n_simulations : int, default=999
        Number of Monte Carlo simulations.
    random_state : int, Generator, or None, default=None
        Controls reproducibility.
    """

    def __init__(
        self,
        detector,
        null_model: BaseNullModel | None = None,
        level: float = 0.05,
        n_simulations: int = 999,
        random_state=None,
    ):
        self.detector = detector
        self.null_model = null_model
        self.level = level
        self.n_simulations = n_simulations
        self.random_state = random_state

    def fit(
        self, X: np.ndarray, y=None, X_calib: np.ndarray | None = None
    ) -> "CalibratedDetector":
        """Fit the detector with calibrated penalties.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to be analysed for changes. The detector is fitted on this
            data, and the penalty is scaled to its length.
        y : None
            Ignored. Present for sklearn API compatibility.
        X_calib : np.ndarray of shape (n_calib, n_features) or None, default=None
            Optional separate null (change-free) dataset used to fit the null
            model. Can be any length. When ``None``, the null model is fitted
            on ``X``. Providing clean null data avoids inflating the null
            distribution with changepoints present in ``X``.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)

        if X_calib is not None:
            X_calib = np.asarray(X_calib, dtype=np.float64)
            if X_calib.ndim != 2:
                raise ValueError(f"`X_calib` must be 2-D, got shape {X_calib.shape}.")
            if X_calib.shape[1] != X.shape[1]:
                raise ValueError(
                    f"`X_calib` has {X_calib.shape[1]} features but `X` has "
                    f"{X.shape[1]}. They must match."
                )

        # Discover PenalisedScore parameters.
        penalised_params = _find_penalised_score_params(self.detector)
        if not penalised_params:
            raise NotImplementedError(
                f"{type(self.detector).__name__} has no top-level PenalisedScore "
                "parameters. CalibratedDetector requires at least one."
            )

        null_model = (
            PermutationNullModel() if self.null_model is None else self.null_model
        )

        # Bonferroni correction across multiple penalties.
        k = len(penalised_params)
        adjusted_level = self.level / k

        # RNG: create a seeded Generator for reproducibility.
        rng = np.random.default_rng(self.random_state)

        calibrated_penalties: dict = {}
        set_params_kwargs: dict = {}

        for param_name, penalised_score in penalised_params.items():
            cal_penalty = calibrate_penalty(
                scorer=penalised_score.scorer,
                X=X,
                null_model=null_model,
                interval_specs=None,
                detector=self.detector,
                level=adjusted_level,
                n_simulations=self.n_simulations,
                random_state=rng,
                X_calib=X_calib,
            )
            calibrated_penalties[param_name] = cal_penalty
            set_params_kwargs[f"{param_name}__penalty"] = cal_penalty

        # Clone detector, inject calibrated penalties, and fit.
        calibrated_detector = clone(self.detector)
        calibrated_detector.set_params(**set_params_kwargs)
        calibrated_detector.fit(X, y)

        self.detector_ = calibrated_detector
        self.calibrated_penalties_ = calibrated_penalties
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
