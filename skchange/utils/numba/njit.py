"""Dispatch njit decorator used to isolate numba."""

from sktime.utils.dependencies import _check_soft_dependencies

from skchange.config import get_config

if _check_soft_dependencies("numba", severity="ignore") and get_config("use_njit"):
    from numba import jit, njit  # noqa E402

    def jit_configured():
        """Jit dectorator with configured arguments."""
        njit_args = get_config("njit_args")
        return jit(**njit_args)

    def njit_configured():
        """Define njit dectorator with configured arguments."""
        njit_args = get_config("njit_args")
        return njit(**njit_args)

else:

    def _identity(func):
        return func

    def jit(*args, **kwargs):
        """Identity decorator for replacing jit by passthrough."""
        return _identity

    def njit(*args, **kwargs):
        """Identity decorator for replacing njit by passthrough."""
        return _identity

    def jit_configured():
        """Identity decorator for replacing configured_jit by passthrough."""
        return _identity

    def njit_configured():
        """Identity decorator for replacing configured_njit by passthrough."""
        return _identity
