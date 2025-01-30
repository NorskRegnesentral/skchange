"""Custom exceptions for skchange."""


class NotAdaptedError(ValueError, AttributeError):
    """Exception class raised if IntervalScorer is evaluated before adapting to data.

    If the IntervalScorer is evaluated on `cuts` only, without providing data,
    the IntervalScorer must be adapted to data previously through the `adapt` method.

    This class inherits from both ``ValueError`` and ``AttributeError`` to help with
    exception handling.

    References
    ----------
    .. [1] Based on sktime's NotFittedError
    """
