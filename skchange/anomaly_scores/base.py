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
