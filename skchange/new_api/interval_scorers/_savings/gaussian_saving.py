"""Gaussian saving for a fixed mean and variance baseline."""

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseSaving
from skchange.new_api.penalties import mvcapa_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit
from skchange.utils.numba.general import truncate_below
from skchange.utils.numba.stats import col_cumsum


@njit
def gaussian_saving(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_var: np.ndarray,
) -> np.ndarray:
    """Calculate the Gaussian saving against a fixed baseline mean and variance.

    The saving is the reduction in negative log-likelihood from using the MLE
    parameters vs. the fixed baseline parameters.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments (inclusive).
    ends : np.ndarray
        End indices of the segments (exclusive).
    sums : np.ndarray
        Cumulative sum of the input data, with a row of 0-entries as the first row.
    sums2 : np.ndarray
        Cumulative sum of the squared input data, with a row of 0-entries as the
        first row.
    baseline_mean : np.ndarray
        Fixed baseline mean of shape (n_features,).
    baseline_var : np.ndarray
        Fixed baseline variance of shape (n_features,).

    Returns
    -------
    savings : np.ndarray
        A 2D array of savings. One row for each interval. The number of columns
        is equal to the number of columns in the input data, where each column
        represents the univariate saving for the corresponding input data column.
    """
    n = (ends - starts).reshape(-1, 1)
    partial_sums = sums[ends] - sums[starts]
    partial_sums2 = sums2[ends] - sums2[starts]

    # MLE cost
    mle_var = partial_sums2 / n - (partial_sums / n) ** 2
    mle_var = truncate_below(mle_var, 1e-16)
    mle_cost = n * np.log(2 * np.pi * mle_var) + n

    # Fixed baseline cost
    quadratic_form = (
        partial_sums2 - 2 * baseline_mean * partial_sums + n * baseline_mean**2
    )
    fixed_cost = n * np.log(2 * np.pi * baseline_var) + quadratic_form / baseline_var

    return fixed_cost - mle_cost


class GaussianSaving(BaseSaving):
    r"""Gaussian saving for a fixed mean and variance baseline.

    The Gaussian saving measures the reduction in negative log-likelihood when
    fitting MLE mean and variance to a segment compared to fixed baseline
    parameters. A large saving for an interval indicates that the baseline
    parameters are a poor fit for the data in that interval.

    .. math::
        S([s, e)) = \ell(X_{s:e}; \hat{\mu}, \hat{\sigma}^2)
            - \ell(X_{s:e}; \mu_0, \sigma_0^2)

    where :math:`\hat{\mu}` and :math:`\hat{\sigma}^2` are the MLE estimates and
    :math:`\mu_0, \sigma_0^2` are the fixed baseline parameters.

    Note that this saving assumes a **fixed baseline** provided at fit time.

    Parameters
    ----------
    baseline_mean : float or array-like of shape (n_features,), default=0.0
        Fixed baseline mean.
    baseline_var : float or array-like of shape (n_features,), default=1.0
        Fixed baseline variance. Must be positive.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import GaussianSaving
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> scorer = GaussianSaving()
    >>> scorer.fit(X)
    GaussianSaving()
    >>> cache = scorer.precompute(X)
    >>> interval_specs = np.array([[0, 50], [50, 100]])
    >>> scorer.evaluate(cache, interval_specs)
    """

    def __init__(
        self,
        baseline_mean: ArrayLike | float = 0.0,
        baseline_var: ArrayLike | float = 1.0,
    ):
        self.baseline_mean = baseline_mean
        self.baseline_var = baseline_var

    @property
    def min_size(self) -> int:
        """Minimum segment size (2, required for MLE variance estimation)."""
        return 2

    def get_default_penalty(self) -> np.ndarray:
        """Get the default penalty for the fitted saving.

        The default penalty is given by
        ``mvcapa_penalty(self.n_samples_in_, self.n_features_in_)``

        Returns
        -------
        np.ndarray of shape (n_features,)
            Default penalty value for each number of affected features.
        """
        check_is_fitted(self)
        # Scaling mvcapa penalty by 1.5 is done to pass the sanity checks in the
        # test suite for CAPA.
        return 1.5 * mvcapa_penalty(self.n_samples_in_, self.n_features_in_, 2)

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit Gaussian saving, validating and broadcasting the baseline parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : GaussianSaving
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)
        n_features = X.shape[1]

        mean = np.asarray(self.baseline_mean, dtype=np.float64)
        if mean.ndim == 0:
            mean = np.full(n_features, mean)
        if mean.shape != (n_features,):
            raise ValueError(
                f"baseline_mean must be a scalar or array of shape "
                f"(n_features,)={(n_features,)}, got shape {mean.shape}."
            )

        var = np.asarray(self.baseline_var, dtype=np.float64)
        if var.ndim == 0:
            var = np.full(n_features, var)
        if var.shape != (n_features,):
            raise ValueError(
                f"baseline_var must be a scalar or array of shape "
                f"(n_features,)={(n_features,)}, got shape {var.shape}."
            )
        if np.any(var <= 0):
            raise ValueError("baseline_var must be strictly positive.")

        self.baseline_mean_ = mean
        self.baseline_var_ = var
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute cumulative sums for efficient interval evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            Dictionary with keys ``"sums"`` and ``"sums2"``: cumulative column sums
            and cumulative column sums of squares, each with a leading row of zeros,
            shape ``(n_samples + 1, n_features)``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        return {
            "sums": col_cumsum(X, init_zero=True),
            "sums2": col_cumsum(X**2, init_zero=True),
        }

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate Gaussian saving on intervals.

        Parameters
        ----------
        cache : dict
            Cache from precompute().
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        savings : ndarray of shape (n_interval_specs, n_features)
            Gaussian savings for each interval and feature.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        return gaussian_saving(
            starts,
            ends,
            cache["sums"],
            cache["sums2"],
            self.baseline_mean_,
            self.baseline_var_,
        )
