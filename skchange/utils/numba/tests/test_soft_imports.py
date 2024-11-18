import os
import sys
from contextlib import contextmanager

import pytest


def remove_modules_with_prefix(prefix):
    to_remove = [mod for mod in sys.modules if mod.startswith(prefix)]
    for mod in to_remove:
        del sys.modules[mod]


@contextmanager
def temp_env_and_modules(remove_module_prefix: str, env_vars: dict):
    original_modules = sys.modules.copy()
    original_environ = os.environ.copy()
    remove_modules_with_prefix(remove_module_prefix)
    os.environ.clear()
    os.environ.update(env_vars)
    try:
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)
        os.environ.clear()
        os.environ.update(original_environ)


def test_setting_wrong_env_variable_raises():
    with (
        temp_env_and_modules(
            remove_module_prefix="skchange", env_vars={"NUMBA_CACHE": "invalid_value"}
        ),
        pytest.raises(ValueError),
    ):
        import skchange.utils.numba  # noqa: F401, I001


def test_setting_truthy_env_variable_does_not_raise():
    with temp_env_and_modules(
        remove_module_prefix="skchange", env_vars={"NUMBA_FASTMATH": "1"}
    ):
        import skchange.utils.numba  # noqa: F401, I001

    assert True


def test_setting_falsy_env_variable_does_not_raise():
    with temp_env_and_modules(
        remove_module_prefix="skchange", env_vars={"NUMBA_FASTMATH": "0"}
    ):
        import skchange.utils.numba  # noqa: F401, I001

    assert True
