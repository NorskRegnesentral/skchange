"""Common validation functions for input parameters."""

from numbers import Number


def check_larger_than(
    min_value: Number, value: Number, name: str, allow_none: bool = False
) -> Number:
    """Check if value is non-negative.

    Parameters
    ----------
    min_value : int, float
        Minimum allowed value.
    value : int, float
        Value to check.
    name : str
        Name of the parameter to be shown in the error message.
    allow_none : bool, optional (default=False)
        Whether to allow None values.

    Returns
    -------
    value : int, float
        Input value.

    Raises
    ------
    ValueError
        If value is negative.
    """
    if not allow_none and value is None:
        raise ValueError(f"{name} cannot be None.")
    if value is not None and value < min_value:
        raise ValueError(f"{name} must be at least {min_value} ({name}={value}).")
    return value


def check_smaller_than(
    max_value: Number, value: Number, name: str, allow_none: bool = False
) -> Number:
    """Check if value is non-negative.

    Parameters
    ----------
    max_value : int, float
        Maximum allowed value.
    value : int, float
        Value to check.
    name : str
        Name of the parameter to be shown in the error message.
    allow_none : bool, optional (default=False)
        Whether to allow None values.

    Returns
    -------
    value : int, float
        Input value.

    Raises
    ------
    ValueError
        If value is negative.
    """
    if not allow_none and value is None:
        raise ValueError(f"{name} cannot be None.")
    if value is not None and value > max_value:
        raise ValueError(f"{name} must be no larger than {max_value} ({name}={value}).")
    return value
