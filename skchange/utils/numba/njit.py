"""Dispatch njit decorator used to isolate numba.

Copied from sktime.utils.numba.njit
"""

from functools import wraps
from os import environ

from sktime.utils.dependencies import _check_soft_dependencies


def define_prange(_):
    """Dispatch prange based on environment variables."""
    if _check_soft_dependencies("numba", severity="none"):
        from numba import prange as numba_prange

        return numba_prange
    else:
        return range


@define_prange
def prange(*args, **kwargs):
    """Dispatch prange based on numba dependency."""
    pass


def read_boolean_env_var(name, default_value):
    """Read a boolean environment variable."""
    truthy_strings = ["", "1", "true", "True", "TRUE"]
    falsy_strings = ["0", "false", "False", "FALSE"]

    env_value = environ.get(name)
    if env_value is None:
        return default_value

    if env_value in truthy_strings:
        return True
    elif env_value in falsy_strings:
        return False
    else:
        raise ValueError(
            f"Invalid value for boolean environment variable '{name}': {env_value}"
        )


def configure_jit(jit_default_kwargs=None):
    """Decorate jit with default kwargs from environment variables."""
    if jit_default_kwargs is None:
        jit_default_kwargs = {}

    def decorator(_):
        if _check_soft_dependencies("numba", severity="none"):
            from numba import jit as numba_jit

            @wraps(numba_jit)
            def jit(maybe_func=None, **kwargs):
                """Dispatch jit decorator based on environment variables."""
                # This syntax overwrites the default kwargs
                # with the provided kwargs if they overlap.
                kwargs = {**jit_default_kwargs, **kwargs}
                return numba_jit(maybe_func, **kwargs)

        else:

            def jit(maybe_func=None, **kwargs):
                """Identity decorator for replacing jit by passthrough."""
                if callable(maybe_func):
                    # Called with the 'direct' syntax:
                    # @jit
                    # def func(*args, **kwargs):
                    #     ...
                    return maybe_func
                else:
                    # Called with arguments to the decorator:
                    # @jit(cache=True)
                    # def func(*args, **kwargs):
                    #     ...
                    def decorator(func):
                        return func

                    return decorator

        return jit

    return decorator


def configure_njit(njit_default_kwargs=None):
    """Configure njit with default kwargs from environment variables."""
    if njit_default_kwargs is None:
        njit_default_kwargs = {}

    def decorator(_):
        if _check_soft_dependencies("numba", severity="none"):
            from numba import njit as numba_njit

            @wraps(numba_njit)
            def njit(maybe_func=None, **kwargs):
                """Dispatch njit decorator based on environment variables."""
                kwargs = {**njit_default_kwargs, **kwargs}
                print(kwargs)
                return numba_njit(maybe_func, **kwargs)

        else:

            def njit(maybe_func=None, **kwargs):
                """Identity decorator for replacing njit by passthrough."""
                if callable(maybe_func):
                    # Called with the 'direct' syntax:
                    # @njit
                    # def func(*args, **kwargs):
                    #     ...
                    return maybe_func
                else:
                    # Called with arguments to the decorator:
                    # @jit(cache=True)
                    # def func(*args, **kwargs):
                    #     ...
                    def decorator(func):
                        return func

                    return decorator

        return njit

    return decorator


@configure_jit(
    jit_default_kwargs={
        "cache": read_boolean_env_var("NUMBA_CACHE", default_value=False),
        "fastmath": read_boolean_env_var("NUMBA_FASTMATH", default_value=False),
        "parallel": read_boolean_env_var("NUMBA_PARALLEL", default_value=False),
    },
)
def jit():
    """Dispatch jit decorator based on environment variables."""
    pass


@configure_njit(
    njit_default_kwargs={
        "cache": read_boolean_env_var("NUMBA_CACHE", default_value=True),
        "fastmath": read_boolean_env_var("NUMBA_FASTMATH", default_value=False),
        "parallel": read_boolean_env_var("NUMBA_PARALLEL", default_value=False),
    }
)
def njit():
    """Dispatch njit decorator based on environment variables."""
    pass
