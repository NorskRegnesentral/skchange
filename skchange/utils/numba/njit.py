"""Dispatch njit decorator used to isolate numba."""

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("numba", severity="ignore"):
    from numba import jit, njit  # noqa E402
else:

    def identity_decorator_factory(*args, **kwargs):
        """Make an identity decorator for replacing jit or njit by passthrough."""

        def decorator(func):
            return func

        return decorator

    jit = identity_decorator_factory
    njit = identity_decorator_factory
