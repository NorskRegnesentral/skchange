"""Configuration module for the package."""

config = {
    "enable_numba": True,
    "njit_args": {
        "cache": True,
        "nogil": False,
        "parallel": False,
    },
}


def get_config(key=None):
    """Get the entire config or a specific key."""
    if key:
        return config.get(key)
    return config


def _update_config(config_updates):
    """Update the configuration with new values."""
    for key, value in config_updates.items():
        if key in config:
            config[key] = value


def update_config(
    enable_numba: bool = None,
    njit_args: dict = None,
):
    """Update the configuration with new values.

    Parameters
    ----------
    enable_numba : bool
        Whether to use njit.
    njit_args : dict
        Arguments to pass to njit.
    """
    config_updates = {}
    if enable_numba is not None:
        config_updates["enable_numba"] = enable_numba
    if njit_args is not None:
        config_updates["njit_args"] = njit_args
    _update_config(config_updates)
