"""The functional pruning optimal partitioning (FPOP) algorithm for change in mean."""

__author__ = ["Tveten"]
__all__ = ["FPOP"]

from numbers import Real

import numpy as np

from skchange.detectors._base import BaseChangeDetector
from skchange.penalties import bic_penalty
from skchange.types import ArrayLike, Self
from skchange.utils import SkchangeTags
from skchange.utils._numba import njit
from skchange.utils._param_validation import Interval, _fit_context
from skchange.utils.validation import check_is_fitted, validate_data


@njit(cache=True)
def _fpop_mean(y: np.ndarray, penalty: float) -> np.ndarray:
    """FPOP algorithm for change in mean of a univariate series.

    Implements the functional pruning optimal partitioning (FPOP) algorithm [1]_
    specialised to the squared-error (L2) cost for changes in the mean.

    The piecewise quadratic functional cost is maintained as a set of pieces
    ordered by their intervals on the mean axis.  Each piece stores the
    quadratic cost f(theta) = A*theta^2 + B*theta + C on its interval.  The
    coefficient A equals the number of data points accumulated in that piece,
    which is used to determine the previous changepoint during backtracking.

    Combined cut-and-add step for data point y_i with level = min + penalty:

    * Portions of the current piecewise function below ``level`` are kept and
      have the new data point added: (A+1, B-2*y, C+y^2).
    * Portions above ``level`` are replaced by a constant piece at ``level``,
      and then have the new data point added: (1, -2*y, level+y^2).

    Parameters
    ----------
    y : np.ndarray of shape (n,)
        Univariate time series.
    penalty : float
        Non-negative penalty per changepoint.

    Returns
    -------
    most_recent_cpts : np.ndarray of shape (n,), dtype int64
        Backtracking array: ``most_recent_changepoints[i]`` is the 0-based index of
        the last observation in the segment preceding the one ending at ``i``, or
        ``-1`` if the entire prefix ``y[0..i]`` is one segment.

    References
    ----------
    .. [1] Maidstone, R., Hocking, T., Rigaill, G., & Fearnhead, P. (2017).
       On optimal multiple changepoint algorithms for large data. Statistics
       and Computing, 27(2), 519-533.
    """
    n = len(y)
    if n == 1:
        return np.empty(0, dtype=np.intp)

    # Upper bound on pieces: each step adds at most one new quadratic piece.
    max_pieces = n + 2

    p_lo = np.empty(max_pieces)
    p_hi = np.empty(max_pieces)
    p_A = np.empty(max_pieces)
    p_B = np.empty(max_pieces)
    p_C = np.empty(max_pieces)

    # Temporary arrays for the new piecewise function built each step.
    t_lo = np.empty(max_pieces)
    t_hi = np.empty(max_pieces)
    t_A = np.empty(max_pieces)
    t_B = np.empty(max_pieces)
    t_C = np.empty(max_pieces)

    # Initialise: one piece covering (-inf, inf) with cost = penalty.
    p_lo[0] = -np.inf
    p_hi[0] = np.inf
    p_A[0] = 0.0
    p_B[0] = 0.0
    p_C[0] = penalty
    n_pieces = 1

    most_recent_changepoints = np.empty(n, dtype=np.int64)

    # --- Step 0: add first data point, no cut needed. ---
    y0 = y[0]
    track_min = np.inf
    track_age = 0
    for k in range(n_pieces):
        p_A[k] += 1.0
        p_B[k] -= 2.0 * y0
        p_C[k] += y0 * y0
        piece_min = -(p_B[k] * p_B[k] / (4.0 * p_A[k])) + p_C[k]
        if piece_min < track_min:
            track_min = piece_min
            track_age = int(p_A[k])
    most_recent_changepoints[0] = 0 - track_age  # = -1 for the initial single piece

    # --- Steps 1 .. n-1: cut-and-add. ---
    for i in range(1, n):
        yi = y[i]
        level = track_min + penalty

        # Build new piecewise function by iterating over old pieces.
        # ``const_start`` tracks the left boundary of the current "constant"
        # region (parts of the domain where the old function >= level).
        n_new = 0
        const_start = -np.inf

        for k in range(n_pieces):
            Ak = p_A[k]
            Bk = p_B[k]
            Ck = p_C[k]

            # Discriminant of Ak*t^2 + Bk*t + Ck = level.
            D = Bk * Bk - 4.0 * Ak * (Ck - level)

            if D <= 0.0:
                # Piece is entirely at or above level; absorbed into constant region.
                continue

            sqrtD = np.sqrt(D)
            two_A = 2.0 * Ak
            r1 = (-Bk - sqrtD) / two_A  # left root
            r2 = (-Bk + sqrtD) / two_A  # right root

            # Clip roots to this piece's interval.
            q_lo = p_lo[k] if p_lo[k] > r1 else r1
            q_hi = p_hi[k] if p_hi[k] < r2 else r2

            if q_lo >= q_hi:
                # Piece interval and below-level region do not overlap.
                continue

            # --- Constant piece from const_start to q_lo (if non-empty). ---
            if const_start < q_lo:
                t_lo[n_new] = const_start
                t_hi[n_new] = q_lo
                t_A[n_new] = 1.0
                t_B[n_new] = -2.0 * yi
                t_C[n_new] = level + yi * yi
                n_new += 1

            # --- Quadratic piece on [q_lo, q_hi] with data point added. ---
            t_lo[n_new] = q_lo
            t_hi[n_new] = q_hi
            t_A[n_new] = Ak + 1.0
            t_B[n_new] = Bk - 2.0 * yi
            t_C[n_new] = Ck + yi * yi
            n_new += 1

            const_start = q_hi

        # --- Trailing constant piece from const_start to +inf. ---
        t_lo[n_new] = const_start
        t_hi[n_new] = np.inf
        t_A[n_new] = 1.0
        t_B[n_new] = -2.0 * yi
        t_C[n_new] = level + yi * yi
        n_new += 1

        # Swap temporary arrays into the main piece arrays.
        n_pieces = n_new
        for k in range(n_pieces):
            p_lo[k] = t_lo[k]
            p_hi[k] = t_hi[k]
            p_A[k] = t_A[k]
            p_B[k] = t_B[k]
            p_C[k] = t_C[k]

        # Find minimum and age of the new piecewise function.
        track_min = np.inf
        for k in range(n_pieces):
            Ak = p_A[k]
            Bk = p_B[k]
            Ck = p_C[k]
            piece_min = -(Bk * Bk / (4.0 * Ak)) + Ck
            if piece_min < track_min:
                track_min = piece_min
                track_age = int(Ak)

        most_recent_changepoints[i] = i - track_age

    return most_recent_changepoints


@njit(cache=True)
def _backtrack(most_recent_changepoints: np.ndarray) -> np.ndarray:
    """Recover changepoints from the FPOP backtracking array.

    Parameters
    ----------
    most_recent_changepoints : np.ndarray of shape (n,), dtype int64
        Backtracking array produced by ``_fpop_mean``.

    Returns
    -------
    changepoints : np.ndarray
        0-based indices of the first observation in each detected segment,
        excluding the implicit first segment starting at 0.
    """
    result = []
    tau = len(most_recent_changepoints) - 1
    while tau >= 0:
        prev = most_recent_changepoints[tau]
        if prev >= 0:
            result.append(prev + 1)
        tau = prev
    # result is in reverse order; reverse before returning.
    n_cpts = len(result)
    out = np.empty(n_cpts, dtype=np.int64)
    for j in range(n_cpts):
        out[j] = result[n_cpts - 1 - j]
    return out


class FPOP(BaseChangeDetector):
    """Functional pruning optimal partitioning (FPOP) for changes in the mean.

    Implements the FPOP algorithm [1]_ specialised to the squared-error (L2)
    cost, detecting changes in the mean of a univariate time series.

    FPOP solves the same penalised optimal partitioning problem as PELT [2]_
    but achieves pruning through a functional representation of the optimal
    cost rather than a scalar threshold.  This typically yields a lower
    computational cost per observation than PELT.

    The algorithm is implemented purely in Numba for speed.

    Parameters
    ----------
    penalty : float or None, default=None
        Penalty incurred for each added changepoint. Must be non-negative.
        If ``None``, defaults to the BIC penalty
        ``2 * log(n_samples)`` after fitting.
    penalty_scale : float, default=1.0
        Multiplicative scale applied to the default BIC penalty when
        ``penalty`` is ``None``.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during ``fit``.
    n_samples_in_ : int
        Number of samples seen during ``fit``.
    penalty_ : float
        Penalty value used.

    References
    ----------
    .. [1] Maidstone, R., Hocking, T., Rigaill, G., & Fearnhead, P. (2017).
       On optimal multiple changepoint algorithms for large data. Statistics
       and Computing, 27(2), 519-533.
    .. [2] Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection
       of changepoints with a linear computational cost. Journal of the American
       Statistical Association, 107(500), 1590-1598.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.detectors import FPOP
    >>> rng = np.random.default_rng(2)
    >>> X = np.concatenate([rng.normal(0, 1, (100, 1)),
    ...                     rng.normal(10, 1, (100, 1))])
    >>> detector = FPOP()
    >>> detector.fit(X).predict(X)
    array([100])

    .. note::
        Only univariate input (``n_features == 1``) is supported.
    """

    _parameter_constraints = {
        "penalty": [Interval(Real, 0, None, closed="left"), None],
        "penalty_scale": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        penalty: float | None = None,
        penalty_scale: float = 1.0,
    ):
        self.penalty = penalty
        self.penalty_scale = penalty_scale

    def __sklearn_tags__(self) -> SkchangeTags:
        tags = super().__sklearn_tags__()
        tags.input_tags.multivariate = False
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit to training data.

        Records ``n_samples_in_``, ``n_features_in_``, and resolves
        ``penalty_``.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training time series.
        y : ignored

        Returns
        -------
        self : FPOP
        """
        X = validate_data(self, X, reset=True, ensure_2d=True)
        if self.n_features_in_ != 1:
            raise ValueError(
                f"FPOP only supports univariate input, got {self.n_features_in_}"
                " features."
            )

        if self.penalty is None:
            base = bic_penalty(self.n_samples_in_)
        else:
            base = self.penalty
        self.penalty_ = self.penalty_scale * base

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Detect changepoints in a time series.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, 1)
            Univariate time series.

        Returns
        -------
        changepoints : np.ndarray of shape (n_changepoints,)
            Sorted 0-based indices of the first observation in each new
            segment.  Empty array if no changepoints are detected.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True)

        changepoints = _fpop_mean(X[:, 0], self.penalty_)
        return _backtrack(changepoints).astype(np.intp)
