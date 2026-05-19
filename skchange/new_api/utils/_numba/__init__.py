"""Numba utility wrappers for skchange new API (private).

Soft-import dispatch for ``jit``, ``njit`` and ``prange``. When ``numba`` is
not installed, identity decorators and the built-in ``range`` are used as
drop-in replacements. No numba defaults are overridden — callers should pass
keyword arguments (e.g. ``@njit(cache=True)``) explicitly when needed.
"""


def _is_numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


numba_available = _is_numba_available()

if numba_available:
    from numba import config as _numba_config

    # The TBB threading layer is not easily available; degrade its priority.
    _numba_config.THREADING_LAYER_PRIORITY = ["omp", "tbb", "workqueue"]

    from numba import jit, njit, prange
else:

    def _identity_decorator(maybe_func=None, **_kwargs):
        """Identity decorator used when numba is not installed.

        Supports both ``@njit`` and ``@njit(...)`` call styles.
        """
        if callable(maybe_func):
            return maybe_func

        def decorator(func):
            return func

        return decorator

    jit = _identity_decorator
    njit = _identity_decorator
    prange = range


from skchange.new_api.utils._numba._helpers import (
    col_cumsum,
    col_median,
    col_repeat,
    compute_finite_difference_derivatives,
    digamma,
    kurtosis,
    log_det_covariance,
    log_gamma,
    row_repeat,
    trigamma,
    truncate_below,
    where,
)

__all__ = [
    "jit",
    "njit",
    "numba_available",
    "prange",
    "col_cumsum",
    "col_median",
    "col_repeat",
    "compute_finite_difference_derivatives",
    "digamma",
    "kurtosis",
    "log_det_covariance",
    "log_gamma",
    "row_repeat",
    "trigamma",
    "truncate_below",
    "where",
]
