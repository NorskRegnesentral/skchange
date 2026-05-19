"""Native L2 transient score for epidemic changepoints."""

__author__ = ["Tveten"]

import numpy as np

from skchange.new_api.interval_scorers._base import BaseTransientScore
from skchange.new_api.penalties import bic_penalty
from skchange.new_api.typing import ArrayLike, Self
from skchange.new_api.utils._numba import col_cumsum, njit
from skchange.new_api.utils._param_validation import _fit_context
from skchange.new_api.utils.validation import (
    check_interval_specs,
    check_is_fitted,
    validate_data,
)


@njit
def l2_transient_score(
    outer_starts: np.ndarray,
    inner_starts: np.ndarray,
    inner_ends: np.ndarray,
    outer_ends: np.ndarray,
    sums: np.ndarray,
) -> np.ndarray:
    """Compute the L2 transient score for each (outer, inner) candidate.

    The transient score is the cost reduction obtained by allowing the inner
    interval ``[inner_start, inner_end)`` to have its own optimal mean,
    relative to the outer interval ``[outer_start, outer_end)`` sharing a
    single mean with the surrounding pieces ``[outer_start, inner_start)``
    and ``[inner_end, outer_end)``::

        score = C(outer) - C(inner) - C(surrounding)

    The sum-of-squares terms cancel exactly in this difference, so only
    cumulative sums of ``X`` (not ``X**2``) are needed:

        score = sum_inner^2 / n_inner + sum_surr^2 / n_surr
              - sum_outer^2 / n_outer
    """
    # Per-segment sample counts.
    n_outer = (outer_ends - outer_starts).reshape(-1, 1)
    n_inner = (inner_ends - inner_starts).reshape(-1, 1)
    n_surr = n_outer - n_inner

    # Per-segment partial sums.
    sum_outer = sums[outer_ends] - sums[outer_starts]
    sum_inner = sums[inner_ends] - sums[inner_starts]
    sum_surr = sum_outer - sum_inner

    # Guard against empty surrounding (n_surr == 0): the surrounding term is 0.
    safe_n_surr = np.where(n_surr > 0, n_surr, 1)
    surr_term = np.where(n_surr > 0, sum_surr**2 / safe_n_surr, 0.0)

    return sum_inner**2 / n_inner + surr_term - sum_outer**2 / n_outer


class L2TransientScore(BaseTransientScore):
    r"""Native L2 transient score (epidemic changepoint model).

    Computes the squared-error reduction from allowing an inner interval
    ``[inner_start, inner_end)`` to have its own optimal mean, relative to
    the surrounding outer interval ``[outer_start, outer_end)`` sharing a
    single mean. Mathematically equivalent to
    :class:`CostTransientScore` wrapping :class:`L2Cost`, but evaluated in a
    single vectorised pass via cumulative sums instead of one inner-loop
    iteration per candidate.

    .. math::
        S(\text{outer}, \text{inner}) =
            C(\text{outer}) - C(\text{inner}) - C(\text{surrounding})

    where the surrounding segment is the concatenation
    ``[outer_start, inner_start) \cup [inner_end, outer_end)``.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import L2TransientScore
    >>> X = np.random.default_rng(0).normal(size=(100, 1))
    >>> scorer = L2TransientScore().fit(X)
    >>> cache = scorer.precompute(X)
    >>> specs = np.array([[0, 40, 50, 100]])
    >>> scorer.evaluate(cache, specs)  # doctest: +ELLIPSIS
    array([[...]])
    """

    _parameter_constraints: dict = {}

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the scorer (stores only ``n_samples_in_`` / ``n_features_in_``)."""
        validate_data(self, X, ensure_2d=True, reset=True)
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute cumulative sums.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            ``{"sums": cumsum(X)}`` with a leading row of zeros.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        return {"sums": col_cumsum(X, init_zero=True)}

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate the L2 transient score on ``[outer_s, inner_s, inner_e, outer_e)``.

        Parameters
        ----------
        cache : dict
            Cache from :meth:`precompute`.
        interval_specs : array-like of shape (n_interval_specs, 4)
            Each row is ``(outer_start, inner_start, inner_end, outer_end)``
            with ``outer_start <= inner_start < inner_end <= outer_end``.

        Returns
        -------
        scores : ndarray of shape (n_interval_specs, n_features)
            L2 transient score per candidate and feature.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            check_sorted=False,
            caller_name=self.__class__.__name__,
        )
        if interval_specs.size == 0:
            return np.empty((0, cache["sums"].shape[1]), dtype=float)

        outer_s = interval_specs[:, 0]
        inner_s = interval_specs[:, 1]
        inner_e = interval_specs[:, 2]
        outer_e = interval_specs[:, 3]
        valid = (outer_s <= inner_s) & (inner_s < inner_e) & (inner_e <= outer_e)
        if not np.all(valid):
            raise ValueError(
                "Each row of `interval_specs` must satisfy "
                "outer_start <= inner_start < inner_end <= outer_end in "
                f"{self.__class__.__name__}."
            )

        return l2_transient_score(outer_s, inner_s, inner_e, outer_e, cache["sums"])

    def get_default_penalty(self) -> float:
        """Return the default BIC penalty (matches :class:`L2Cost`)."""
        check_is_fitted(self)
        return bic_penalty(self.n_samples_in_, self.n_features_in_)
