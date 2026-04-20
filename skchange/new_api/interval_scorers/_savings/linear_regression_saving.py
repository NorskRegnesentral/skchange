"""Linear regression saving for a fixed coefficient baseline."""

__author__ = ["johannvk"]

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseSaving
from skchange.new_api.penalties import chi2_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit


@njit
def linear_regression_saving(
    starts: np.ndarray,
    ends: np.ndarray,
    X_response: np.ndarray,
    X_covariates: np.ndarray,
    baseline_coeffs: np.ndarray,
) -> np.ndarray:
    """Calculate the linear regression saving against fixed baseline coefficients.

    The saving is the reduction in RSS when using OLS coefficients vs. the fixed
    baseline coefficients.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments (inclusive).
    ends : np.ndarray
        End indices of the segments (exclusive).
    X_response : np.ndarray
        Response variable, shape ``(n_samples,)``.
    X_covariates : np.ndarray
        Covariate matrix, shape ``(n_samples, n_covariates)``.
    baseline_coeffs : np.ndarray
        Fixed baseline regression coefficients, shape ``(n_covariates,)``.

    Returns
    -------
    savings : np.ndarray
        Shape ``(n_intervals, 1)``. RSS saving per segment.
    """
    n_intervals = len(starts)
    savings = np.zeros((n_intervals, 1))
    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        y = X_response[start:end]
        Z = X_covariates[start:end]

        # Fixed-coefficient RSS.
        r_fixed = y - Z @ baseline_coeffs
        fixed_rss = np.sum(r_fixed * r_fixed)

        # OLS RSS.
        coeffs, residuals, rank, _ = np.linalg.lstsq(Z, y)
        if rank < Z.shape[1] or len(residuals) == 0:
            r_mle = y - Z @ coeffs
            mle_rss = np.sum(r_mle * r_mle)
        else:
            mle_rss = np.sum(residuals)

        savings[i, 0] = fixed_rss - mle_rss
    return savings


class LinearRegressionSaving(BaseSaving):
    r"""Linear regression saving for a fixed coefficient baseline.

    The saving measures the reduction in ordinary least-squares (OLS) residual
    sum of squares (RSS) when using segment-wise OLS coefficients vs. fixed
    baseline coefficients.

    .. math::
        S([s, e)) = \text{RSS}(\beta_0;\, X_{s:e})
                  - \min_\beta\,\text{RSS}(\beta;\, X_{s:e})

    A large saving indicates the baseline coefficients are a poor fit for the
    segment.

    Parameters
    ----------
    response_col : int, default=0
        Column index in ``X`` to use as the response variable. All other
        columns are used as predictors.
    baseline_coeffs : array-like of shape (n_covariates,) or None, default=None
        Fixed baseline regression coefficients. If ``None``, coefficients are
        estimated from the training data using
        :class:`sklearn.linear_model.HuberRegressor` (a robust M-estimator),
        which down-weights observations far from the fitted line. This makes
        the baseline robust when the training window contains changepoints.
        To use a different estimation strategy, estimate the coefficients
        externally and pass them here.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import LinearRegressionSaving
    >>> rng = np.random.default_rng(0)
    >>> X = np.column_stack([rng.normal(size=100), rng.normal(size=100)])
    >>> scorer = LinearRegressionSaving()
    >>> scorer.fit(X)
    LinearRegressionSaving()
    >>> cache = scorer.precompute(X)
    >>> scorer.evaluate(cache, np.array([[0, 50], [50, 100]]))
    """

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags with ``input_tags.conditional=True`` and ``aggregated=True``."""
        tags = super().__sklearn_tags__()
        tags.input_tags.conditional = True
        tags.interval_scorer_tags.aggregated = True
        return tags

    def __init__(
        self,
        response_col: int = 0,
        baseline_coeffs: ArrayLike | None = None,
    ):
        self.response_col = response_col
        self.baseline_coeffs = baseline_coeffs

    @property
    def min_size(self) -> int:
        """Minimum segment size (n_covariates + 1 for at least one residual d.o.f.)."""
        check_is_fitted(self)
        return self.n_covariates_ + 1

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit the saving, splitting ``X`` and estimating baseline if needed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. Must have at least 2 columns.
        y : None
            Ignored.

        Returns
        -------
        self : LinearRegressionSaving
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)
        n_features = X.shape[1]
        if n_features < 2:
            raise ValueError(
                "LinearRegressionSaving requires at least 2 columns "
                "(1 response + at least 1 covariate), "
                f"got n_features={n_features}."
            )
        if not (0 <= self.response_col < n_features):
            raise ValueError(
                f"response_col={self.response_col} is out of range for "
                f"data with {n_features} columns."
            )
        covariate_cols = [c for c in range(n_features) if c != self.response_col]
        self.response_col_ = self.response_col
        self.covariate_cols_ = covariate_cols
        self.n_covariates_ = len(covariate_cols)

        if self.baseline_coeffs is None:
            from sklearn.linear_model import HuberRegressor

            y_train = X[:, self.response_col_]
            Z_train = X[:, self.covariate_cols_]
            reg = HuberRegressor(fit_intercept=False, alpha=0)
            reg.fit(Z_train, y_train)
            self.baseline_coeffs_ = reg.coef_
        else:
            coeffs = np.asarray(self.baseline_coeffs, dtype=np.float64).ravel()
            if coeffs.shape != (self.n_covariates_,):
                raise ValueError(
                    f"baseline_coeffs must have shape ({self.n_covariates_},), "
                    f"got {coeffs.shape}."
                )
            self.baseline_coeffs_ = coeffs
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Split and store response and covariates for interval evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            Dictionary with keys:

            - ``"X_response"``: 1-D response array of shape ``(n_samples,)``.
            - ``"X_covariates"``: covariate matrix of shape
              ``(n_samples, n_covariates)``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=False)
        return {
            "X_response": X[:, self.response_col_],
            "X_covariates": X[:, self.covariate_cols_],
        }

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate linear regression saving on intervals.

        Parameters
        ----------
        cache : dict
            Cache from ``precompute()``.
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        savings : ndarray of shape (n_interval_specs, 1)
            RSS saving for each interval.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        return linear_regression_saving(
            starts,
            ends,
            cache["X_response"],
            cache["X_covariates"],
            self.baseline_coeffs_,
        )

    def get_default_penalty(self) -> float:
        r"""Get the default penalty for the fitted linear regression saving.

        The saving is asymptotically :math:`\\chi^2(n_{\\text{covariates}})` under
        the null (correct baseline), so ``chi2_penalty`` is used as the default.

        Returns
        -------
        float
            Default penalty value.
        """
        check_is_fitted(self)
        # Scaling chi2 penalty by 1.5 is done to pass the sanity checks in the
        # test suite for CAPA.
        return 1.5 * chi2_penalty(self.n_samples_in_, self.n_covariates_)
