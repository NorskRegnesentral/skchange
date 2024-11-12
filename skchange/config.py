"""Configuration module for the package."""

config = {
    "use_njit": True,
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
    use_njit: bool = None,
    njit_args: dict = None,
):
    """Update the configuration with new values.

    Parameters
    ----------
    use_njit : bool
        Whether to use njit.
    njit_args : dict
        Arguments to pass to njit.
    """
    config_updates = {}
    if use_njit is not None:
        config_updates["use_njit"] = use_njit
    if njit_args is not None:
        config_updates["njit_args"] = njit_args
    _update_config(config_updates)
