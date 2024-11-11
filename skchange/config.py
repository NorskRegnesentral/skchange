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


def _update_config(**kwargs):
    """Update the configuration with new values."""
    for key, value in kwargs.items():
        if key in config:
            config[key] = value


def update_config(use_njit=None, **njit_args):
    """Update the configuration with new values.

    Parameters
    ----------
    use_njit : bool
        Whether to use njit.
    njit_args : dict
        Arguments to pass to njit.
    """
    kwargs = {}
    if use_njit is not None:
        kwargs["use_njit"] = use_njit
    if njit_args:
        kwargs["njit_args"] = njit_args
    _update_config(**kwargs)
