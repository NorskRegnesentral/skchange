"""Dispatch njit decorator used to isolate numba.

Copied from sktime.utils.numba.njit
"""

from functools import wraps
from os import environ

from sktime.utils.dependencies import _check_soft_dependencies


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


def define_jit_and_njit():
    # exports numba.njit if numba is present, otherwise an identity njit
    if _check_soft_dependencies("numba", severity="ignore"):
        from numba import jit as numba_jit, njit as numba_njit  # noqa E402

        # Read possible 'NUMBA_CACHE' environment variable:
        numba_cache = read_boolean_env_var("NUMBA_CACHE", default_value=True)

        # Read possible 'NUMBA_PARALLEL' environment variable:
        numba_parallel = read_boolean_env_var("NUMBA_PARALLEL", default_value=False)

        # Read possible 'NUMBA_FASTMATH' environment variable:
        numba_fastmath = read_boolean_env_var("NUMBA_FASTMATH", default_value=False)

        def jit(**kwargs):
            """Dispatch jit decorator based on environment variables."""
            if "cache" not in kwargs:
                kwargs["cache"] = numba_cache

            if "fastmath" not in kwargs:
                kwargs["fastmath"] = numba_fastmath

            if "parallel" not in kwargs:
                kwargs["parallel"] = numba_parallel

            return numba_jit(**kwargs)

        # Figure out how to use "wraps" correctly here:
        def njit(**kwargs):
            """Dispatch njit decorator based on environment variables."""
            if "cache" not in kwargs:
                kwargs["cache"] = numba_cache

            if "fastmath" not in kwargs:
                kwargs["fastmath"] = numba_fastmath

            if "parallel" not in kwargs:
                kwargs["parallel"] = numba_parallel

            return numba_njit(**kwargs)

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

    return jit, njit


jit, njit = define_jit_and_njit()
