"""Linear regression (sum of squared residuals) cost function."""

__author__ = ["johannvk"]

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseCost
from skchange.new_api.penalties import bic_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit


@njit
def linear_regression_cost(
    starts: np.ndarray,
    ends: np.ndarray,
    X_response: np.ndarray,
    X_covariates: np.ndarray,
) -> np.ndarray:
    """Calculate the linear regression cost (RSS) for each segment.

    Fits OLS coefficients per segment and returns the sum of squared residuals.

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

    Returns
    -------
    costs : np.ndarray
        Shape ``(n_intervals, 1)``. Sum of squared residuals per segment.
    """
    n_intervals = len(starts)
    costs = np.zeros((n_intervals, 1))
    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        y = X_response[start:end]
        Z = X_covariates[start:end]
        coeffs, residuals, rank, _ = np.linalg.lstsq(Z, y)
        if rank < Z.shape[1] or len(residuals) == 0:
            # Underdetermined or degenerate: compute manually.
            r = y - Z @ coeffs
            costs[i, 0] = np.sum(r * r)
        else:
            costs[i, 0] = np.sum(residuals)
    return costs


class LinearRegressionCost(BaseCost):
    r"""Linear regression sum-of-squared-residuals cost.

    Computes the ordinary least-squares (OLS) residual sum of squares (RSS)
    for each candidate segment. One column of the input data ``X`` is used as
    the response variable; the remaining columns are used as predictors.

    .. math::
        C(X_{s:e}) = \min_{\beta}\,\sum_{i=s}^{e-1}
        \bigl(y_i - z_i^\top \beta\bigr)^2

    where :math:`y_i` is the response and :math:`z_i` the covariate vector.

    Parameters
    ----------
    response_col : int, default=0
        Column index in ``X`` to use as the response variable. All other
        columns are used as predictors (intercept not added automatically).

    Notes
    -----
    Requires ``input_tags.no_validation = False``. Because the scorer is
    conditional (uses covariates), at least two input columns are needed.
    ``min_size`` equals the number of covariate columns after fitting.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import LinearRegressionCost
    >>> rng = np.random.default_rng(0)
    >>> X = np.column_stack([rng.normal(size=100), rng.normal(size=100)])
    >>> cost = LinearRegressionCost()
    >>> cost.fit(X)
    LinearRegressionCost()
    >>> cache = cost.precompute(X)
    >>> cost.evaluate(cache, np.array([[0, 50], [50, 100]]))
    """

    def __init__(self, response_col: int = 0):
        self.response_col = response_col

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags with ``input_tags.conditional=True`` and ``aggregated=True``."""
        tags = super().__sklearn_tags__()
        tags.input_tags.conditional = True
        tags.interval_scorer_tags.aggregated = True
        return tags

    @property
    def min_size(self) -> int:
        """Minimum segment size (n_covariates + 1 for at least one residual d.o.f.)."""
        check_is_fitted(self)
        return self.n_covariates_ + 1

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit the cost, splitting ``X`` into response and covariate arrays.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. Must have at least 2 columns.
        y : None
            Ignored.

        Returns
        -------
        self : LinearRegressionCost
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)
        n_features = X.shape[1]
        if n_features < 2:
            raise ValueError(
                "LinearRegressionCost requires at least 2 columns "
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
        """Evaluate linear regression cost on intervals.

        Parameters
        ----------
        cache : dict
            Cache from ``precompute()``.
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        costs : ndarray of shape (n_interval_specs, 1)
            OLS RSS for each interval.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        return linear_regression_cost(
            starts, ends, cache["X_response"], cache["X_covariates"]
        )

    def get_default_penalty(self) -> float:
        """Get the default BIC penalty for the fitted linear regression cost.

        The cost estimates ``n_covariates`` regression coefficients per segment.

        Returns
        -------
        float
            Default BIC penalty value.
        """
        check_is_fitted(self)
        return bic_penalty(self.n_samples_in_, self.n_covariates_)
