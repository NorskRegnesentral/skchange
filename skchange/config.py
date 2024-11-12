"""Configuration module for the package."""


class Config:
    """Configuration class for skchange."""

    def __init__(self):
        """Initialize the default configuration."""
        self._enable_numba = True
        self._njit_args = {
            "cache": True,
            "nogil": False,
            "parallel": False,
        }

    @property
    def enable_numba(self):
        """Enable numba for just-in-time compilation."""
        return self._enable_numba

    @enable_numba.setter
    def enable_numba(self, value):
        if not isinstance(value, bool):
            raise ValueError("enable_numba must be a boolean")
        self._enable_numba = value

    @property
    def njit_args(self):
        """Arguments for numba's njit decorator."""
        return self._njit_args

    @njit_args.setter
    def njit_args(self, value):
        if not isinstance(value, dict):
            raise ValueError("njit_args must be a dictionary")
        self._njit_args = value

    def get(self, key=None):
        """Get the entire config or a specific key."""
        if key:
            return getattr(self, key, None)
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("_")
        }


config = Config()
