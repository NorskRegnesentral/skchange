"""Dispatch njit decorator used to isolate numba."""

from sktime.utils.dependencies import _check_soft_dependencies

from skchange.config import config

if _check_soft_dependencies("numba", severity="ignore") and config.enable_numba:
    from numba import jit, njit  # noqa E402

    jit_configured = jit(**config.njit_args)
    njit_configured = njit(**config.njit_args)

else:

    def _identity(func):
        return func

    def jit(*args, **kwargs):
        """Identity decorator for replacing jit by passthrough."""
        return _identity

    def njit(*args, **kwargs):
        """Identity decorator for replacing njit by passthrough."""
        return _identity

    def jit_configured(*args, **kwargs):
        """Identity decorator for replacing jit_configured by passthrough."""
        return _identity

    def njit_configured(*args, **kwargs):
        """Identity decorator for replacing njit_configured by passthrough."""
        return _identity
