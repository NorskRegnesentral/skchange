"""Base classes for anomaly scores."""

from skchange.base import BaseIntervalEvaluator


class BaseSaving(BaseIntervalEvaluator):
    """Base class template for savings.

    A saving is a measure of the difference between a cost with a fixed baseline
    parameter and an optimised cost over an interval. Most commonly, the baseline
    parameter is pre-calculated robustly over the entire dataset under the assumption
    that anomalies are rare. Each saving thus represents the potential cost reduction if
    the parameter was optimised for the interval.
    """

    expected_interval_entries = 2

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_param_size(p: int) -> int:
        """Get the number of parameters in the saving function.

        Defaults to 1 parameter per variable in the data. This method should be
        overwritten in subclasses if the cost function has a different number of
        parameters per variable.

        Parameters
        ----------
        p : int
            Number of variables in the data.
        """
        return p


class BaseLocalAnomalyScore(BaseIntervalEvaluator):
    """Base class template for local anomaly scores.

    Local anomaly scores are used to detect anomalies in a time series or sequence by
    evaluating the deviation of the data distribution within a subinterval of a larger,
    local interval.
    """

    expected_interval_entries = 4

    def __init__(self):
        super().__init__()
