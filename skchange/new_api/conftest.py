"""Shared pytest fixtures and test data helpers for ``skchange.new_api`` tests."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone

_N_SAMPLES = 100
CHANGEPOINT = 75


def _make_regression_X(
    n_samples: int,
    n_features: int,
    coef_before: np.ndarray,
    coef_after: np.ndarray,
    changepoint: int,
    noise_std: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Generate regression data with a coefficient change at ``changepoint``.

    Column 0 is the response variable ``y``; columns 1+ are the covariates ``Z``.
    The response is generated as ``y = Z @ coef + noise``, where the coefficient
    vector changes from ``coef_before`` to ``coef_after`` at ``changepoint``.

    Parameters
    ----------
    n_samples : int
        Total number of observations.
    n_features : int
        Total number of columns in X (response + covariates).
        ``n_covariates = n_features - 1``.
    coef_before : np.ndarray of shape (n_covariates,)
        Regression coefficients for the first segment ``[0, changepoint)``.
    coef_after : np.ndarray of shape (n_covariates,)
        Regression coefficients for the second segment ``[changepoint, n_samples)``.
    changepoint : int
        Index of the first sample belonging to the second regime.
    noise_std : float, default=0.5
        Standard deviation of the Gaussian noise added to the response.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Combined data matrix with response in column 0 and covariates in columns 1+.
    """
    rng = np.random.default_rng(seed)
    n_covariates = n_features - 1
    Z = rng.normal(size=(n_samples, n_covariates))
    noise = rng.normal(scale=noise_std, size=n_samples)
    y = np.empty(n_samples)
    y[:changepoint] = Z[:changepoint] @ coef_before + noise[:changepoint]
    y[changepoint:] = Z[changepoint:] @ coef_after + noise[changepoint:]
    return np.column_stack([y, Z])


def _make_kink_X(
    n_samples: int,
    n_features: int,
    changepoint: int,
    slope_before: float = 0.0,
    slope_after: float = 1.0,
    noise_std: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    """Generate piecewise-linear data with a change in slope (kink) at ``changepoint``.

    The signal is continuous: both segments share the same value at ``changepoint``.

    Parameters
    ----------
    n_samples : int
        Total number of observations.
    n_features : int
        Number of independent columns.
    changepoint : int
        Index of the first sample in the second segment.
    slope_before : float, default=0.0
        Slope of the first segment.
    slope_after : float, default=1.0
        Slope of the second segment.
    noise_std : float, default=0.2
        Standard deviation of added Gaussian noise.
    seed : int, default=0
        Random seed.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=float)
    kink_value = slope_before * changepoint
    signal = np.where(
        t < changepoint,
        slope_before * t,
        kink_value + slope_after * (t - changepoint),
    )
    noise = rng.normal(scale=noise_std, size=(n_samples, n_features))
    return signal[:, np.newaxis] + noise


def make_single_change_X(
    estimator: BaseEstimator,
    n_features: int | None = None,
    *,
    loc_before: float = 0.0,
    loc_after: float = 10.0,
    scale: float = 1.0,
    poisson_rate_before: float = 2.0,
    poisson_rate_after: float = 10.0,
    regression_coef_before: float = 1.0,
    regression_coef_after: float = 5.0,
    linear_trend_slope_before: float = 0.0,
    linear_trend_slope_after: float = 1.0,
    seed: int = 51,
) -> np.ndarray:
    """Generate test data appropriate for the given estimator based on its tags.

    Parameters
    ----------
    estimator : BaseEstimator
        An estimator instance whose tags determine the generated data shape and type.
    n_features : int or None, default=None
        Number of columns in the returned array. If None, inferred from estimator
        tags. For conditional estimators, ``n_features`` includes 1 response column
        and ``n_features - 1`` covariate columns; defaults to 2. For all other
        estimators, defaults to 1.
    loc_before : float, default=0.0
        Mean of the first segment for Gaussian data.
    loc_after : float, default=10.0
        Mean of the second segment for Gaussian data.
    scale : float, default=1.0
        Standard deviation for Gaussian data.
    poisson_rate_before : float, default=2.0
        Poisson rate for the first segment (``input_tags.integer_only=True``).
    poisson_rate_after : float, default=10.0
        Poisson rate for the second segment.
    regression_coef_before : float, default=1.0
        Scalar coefficient broadcast to all covariates in the first segment.
    regression_coef_after : float, default=5.0
        Scalar coefficient broadcast to all covariates in the second segment.
    linear_trend_slope_before : float, default=0.0
        Slope of the linear trend in the first segment.
    linear_trend_slope_after : float, default=1.0
        Slope of the linear trend in the second segment.
    seed : int, default=51
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Two-segment data with a structural break at sample ``CHANGEPOINT``.
        Signal type is chosen based on estimator tags:

        - ``interval_scorer_tags.linear_trend_segment=True`` or
          ``change_detector_tags.linear_trend_segment=True``: Piecewise linear signal
          with a kink (change in slope) at ``CHANGEPOINT``.
        - ``input_tags.conditional=True``: Linear regression data with a coefficient
          shift. Column 0 is the response; columns 1+ are covariates.
        - ``input_tags.integer_only=True``: Poisson count data with a rate shift.
        - ``input_tags.integer_only=False``: Gaussian data with a mean shift.

        When ``input_tags.timestamps=True``, one column is reserved for monotonically
        increasing integer time steps (at index ``estimator.time_col``). The remaining
        ``n_features - 1`` columns carry the signal determined above. The timestamp
        column is inserted last, after the signal is generated.
    """
    tags = estimator.__sklearn_tags__()
    input_tags = tags.input_tags

    if n_features is None:
        n_features = 2 if (input_tags.conditional or input_tags.timestamps) else 1

    n_signal = n_features - 1 if input_tags.timestamps else n_features

    scorer_tags = tags.interval_scorer_tags
    detector_tags = tags.change_detector_tags
    linear_trend_segment = (
        scorer_tags is not None and scorer_tags.linear_trend_segment
    ) or (detector_tags is not None and detector_tags.linear_trend_segment)

    if linear_trend_segment:
        data = _make_kink_X(
            _N_SAMPLES,
            n_signal,
            CHANGEPOINT,
            slope_before=linear_trend_slope_before,
            slope_after=linear_trend_slope_after,
            seed=seed,
        )
    elif input_tags.conditional:
        n_covariates = n_signal - 1
        coef_before = regression_coef_before * np.ones(n_covariates)
        coef_after = regression_coef_after * np.ones(n_covariates)
        data = _make_regression_X(
            _N_SAMPLES, n_signal, coef_before, coef_after, CHANGEPOINT, seed=seed
        )
    else:
        rng = np.random.default_rng(seed)
        if input_tags.integer_only:
            data = np.concatenate(
                [
                    rng.poisson(lam=poisson_rate_before, size=(CHANGEPOINT, n_signal)),
                    rng.poisson(
                        lam=poisson_rate_after,
                        size=(_N_SAMPLES - CHANGEPOINT, n_signal),
                    ),
                ]
            ).astype(float)
        else:
            data = np.concatenate(
                [
                    rng.normal(
                        loc=loc_before, scale=scale, size=(CHANGEPOINT, n_signal)
                    ),
                    rng.normal(
                        loc=loc_after,
                        scale=scale,
                        size=(_N_SAMPLES - CHANGEPOINT, n_signal),
                    ),
                ]
            )

    if input_tags.timestamps:
        time_col = getattr(estimator, "time_col", 0)
        t = np.arange(_N_SAMPLES, dtype=float)
        return np.insert(data, time_col, t, axis=1)

    return data


def make_no_change_X(
    estimator: BaseEstimator,
    n_features: int | None = None,
    *,
    loc: float = 0.0,
    scale: float = 1.0,
    poisson_rate: float = 5.0,
    regression_coef: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate stationary test data (no structural break) for the given estimator.

    Parameters
    ----------
    estimator : BaseEstimator
        An estimator instance whose tags determine the generated data shape and type.
    n_features : int or None, default=None
        Number of columns in the returned array. If None, inferred from estimator
        tags. For conditional estimators, ``n_features`` includes 1 response column
        and ``n_features - 1`` covariate columns; defaults to 2. For all other
        estimators, defaults to 1.
    loc : float, default=0.0
        Mean for Gaussian data.
    scale : float, default=1.0
        Standard deviation for Gaussian data.
    poisson_rate : float, default=5.0
        Poisson rate for integer-only data.
    regression_coef : float, default=1.0
        Scalar coefficient broadcast to all covariates.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Homogeneous data drawn from a single distribution throughout.
        Signal type is chosen based on estimator tags:

        - ``input_tags.conditional=True``: Linear regression data with constant
          coefficients. Column 0 is the response; columns 1+ are covariates.
        - ``input_tags.integer_only=True``: Poisson count data with constant rate.
        - ``input_tags.integer_only=False``: Gaussian data with constant mean.

        When ``input_tags.timestamps=True``, one column is reserved for monotonically
        increasing integer time steps (at index ``estimator.time_col``). The remaining
        ``n_features - 1`` columns carry the signal determined above. The timestamp
        column is inserted last, after the signal is generated.
    """
    tags = estimator.__sklearn_tags__()
    input_tags = tags.input_tags

    if n_features is None:
        n_features = 2 if (input_tags.conditional or input_tags.timestamps) else 1

    n_signal = n_features - 1 if input_tags.timestamps else n_features

    if input_tags.conditional:
        n_covariates = n_signal - 1
        coef = regression_coef * np.ones(n_covariates)
        data = _make_regression_X(
            _N_SAMPLES, n_signal, coef, coef, _N_SAMPLES, seed=seed
        )
    else:
        rng = np.random.default_rng(seed)
        if input_tags.integer_only:
            data = rng.poisson(lam=poisson_rate, size=(_N_SAMPLES, n_signal)).astype(
                float
            )
        else:
            data = rng.normal(loc=loc, scale=scale, size=(_N_SAMPLES, n_signal))

    if input_tags.timestamps:
        time_col = getattr(estimator, "time_col", 0)
        t = np.arange(_N_SAMPLES, dtype=float)
        return np.insert(data, time_col, t, axis=1)

    return data


@pytest.fixture
def estimator(request):
    """Clone a fresh unfitted estimator from a registry instance."""
    return clone(request.param)
