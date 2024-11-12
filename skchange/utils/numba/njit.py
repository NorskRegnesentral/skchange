"""Dispatch njit decorator used to isolate numba.

Copied from sktime.utils.numba.njit
"""

from sktime.utils.dependencies import _check_soft_dependencies

# exports numba.njit if numba is present, otherwise an identity njit
if _check_soft_dependencies("numba", severity="ignore"):
    from numba import jit, njit  # noqa E402
else:

    def jit(*args, **kwargs):
        """Identity decorator for replacing njit by passthrough."""

        def decorator(func):
            return func

        return decorator

    def njit(*args, **kwargs):
        """Identity decorator for replacing njit by passthrough."""

        def decorator(func):
            return func

        return decorator
