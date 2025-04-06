"""Implementation of the CROPS algorithm for path solutions to penalized CPD."""

import numpy as np

from . import PELT


def pelt_crops(
    change_point_detector: PELT,
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
    pass
