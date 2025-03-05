"""Base classes for anomaly scores."""

from skchange.base import BaseIntervalScorer


class BaseSaving(BaseIntervalScorer):
    """Base class template for savings.

    A saving is a measure of the difference between a cost with a fixed baseline
    parameter and an optimised cost over an interval. Most commonly, the baseline
    parameter is pre-calculated robustly over the entire dataset under the assumption
    that anomalies are rare. Each saving thus represents the potential cost reduction if
    the parameter was optimised for the interval.
    """

    expected_cut_entries = 2

    def __init__(self):
        super().__init__()


class BaseLocalAnomalyScore(BaseIntervalScorer):
    """Base class template for local anomaly scores.

    Local anomaly scores are used to detect anomalies in a time series or sequence by
    evaluating the deviation of the data distribution within a subinterval of a larger,
    local interval.
    """

    expected_cut_entries = 4

    def __init__(self):
        super().__init__()
