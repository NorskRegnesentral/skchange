"""Implementation of the CROPS algorithm for path solutions to penalized CPD."""

import numpy as np

from ..costs.base import BaseCost
from ..penalties import ConstantPenalty
from . import PELT


def pelt_crops(
    cost: BaseCost,
    X: np.ndarray,
    min_penalty: float,
    max_penalty: float,
):
    """Run the CROPS algorithm for path solutions to penalized CPD.

    Reference: https://arxiv.org/pdf/1412.3617

    Parameters
    ----------
    change_point_detector : PELT
        The change point detector to use.
    min_penalty : float
        The minimum penalty to use.
    max_penalty : float
        The maximum penalty to use.
    """
    # Want a Heap of some kind, so that I can pop
    # and push intervals onto it.
    penalty_intervals = [
        (min_penalty, max_penalty),
    ]
    change_point_detector = PELT(cost=cost)
    change_point_detector.fit(X)
    while len(penalty_intervals) > 0:
        # Pop the interval with the lowest penalty.
        min_penalty, max_penalty = penalty_intervals.pop(0)

        # Run the change point detector on the interval.
        # Need to update penalty for the PELT-detector:
        min_penalty_change_points = change_point_detector.set_params(
            penalty=min_penalty
        ).predict(X)
        max_penalty_change_points = change_point_detector.set_params(
            penalty=max_penalty
        ).predict(X)

    pass
