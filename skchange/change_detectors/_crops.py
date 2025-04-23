"""Implementation of the CROPS algorithm for path solutions to penalized CPD."""

import numpy as np
import pandas as pd

from ..change_detectors._pelt import (
    run_improved_pelt_array_based,
    run_pelt_array_based,
    run_pelt_masked,
)
from ..costs.base import BaseCost
from ..penalties import ConstantPenalty
from . import PELT
from .base import BaseChangeDetector


def format_crops_results(
    penalty_to_solution_dict: dict[float, (np.ndarray, float)],
) -> pd.DataFrame:
    """Format the CROPS results into a DataFrame.

    Parameters
    ----------
    cpd_results : dict
        Dictionary with penalty values as keys and tuples of change points and
        segmentation costs as values.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        DataFrame with columns 'num_change_points', 'penalty', and 'segmentation_cost'.
        Dictionary with number of change points as keys and change points as values.
    """
    # Convert the dict lookup results to a list of tuples:
    list_cpd_results = [
        (len(change_points), penalty, segmentation_cost, change_points)
        for (
            penalty,
            (change_points, segmentation_cost),
        ) in penalty_to_solution_dict.items()
    ]

    # When different penalties give the same number of change points, we want to
    # keep the segmentation with the lowest penalty. So we sort by number of
    # change points, and then negative penalty, in reverse order.
    list_cpd_results.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    # Remove duplicates, and keep the first one:
    encountered_num_change_points = set()
    unique_cpd_results = []
    for i in range(len(list_cpd_results)):
        num_change_points = list_cpd_results[i][0]
        if num_change_points in encountered_num_change_points:
            continue
        else:
            unique_cpd_results.append(list_cpd_results[i])
            encountered_num_change_points.add(num_change_points)

    # Extract out the change points into a dict with the number of change points
    # as the key, and the change points as the value.
    change_points_dict = {len(x[3]): x[3] for x in unique_cpd_results}

    # Extract out the change points metadata into a DataFrame.
    change_points_metadata = [x[0:3] for x in unique_cpd_results]
    penalty_change_point_metadata_df = pd.DataFrame(
        change_points_metadata,
        columns=[
            "num_change_points",
            "penalty",
            "segmentation_cost",
        ],
    )

    return penalty_change_point_metadata_df, change_points_dict


class CROPS_PELT(BaseChangeDetector):
    """CROPS algorithm for path solutions to penalized CPD.

    The algorithm can fail when the penalty is low, and the number of
    change points is high. In such cases, it can help to increase the
    `percent_pruning_margin` parameter, increasing the `min_penalty`,
    or increasing the `min_segment_length`.
    If these changes do not help, one can set `drop_pruning` to True,
    which will revert to the optimal partitioning algorithm.
    This will be slower, but can be useful for debugging and testing.

    Reference: https://arxiv.org/pdf/1412.3617

    Parameters
    ----------
    cost : BaseCost
        The cost function to use.
    min_penalty : float
        The minimum penalty to use.
    max_penalty : float
        The maximum penalty to use.
    min_segment_length : int, default=1
        The minimum segment length to use.
    split_cost : float, optional, default=0.0
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the above inequality.
    percent_pruning_margin : float, optional, default=0.1
        The percentage of pruning margin to use. By default set to 0.0.
    middle_penalty_nudge : float, optional, default=1.0e-4
        Size of multiplicative nudge to apply to the middle penalty
        value to avoid numerical instability. By default set to 1.0e-4.
    drop_pruning: bool, optional
        If True, drop the pruning step. Reverts to optimal partitioning.
        Can be useful for debugging and testing.  By default set to False.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": ["johannvk"],
        "fit_is_empty": True,
    }

    def __init__(
        self,
        cost: BaseCost,
        min_penalty: float,
        max_penalty: float,
        min_segment_length: int = 1,
        split_cost: float = 0.0,
        percent_pruning_margin: float = 0.0,
        middle_penalty_nudge: float = 1.0e-4,
        drop_pruning: bool = False,
    ):
        super().__init__()
        self.cost = cost
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.min_segment_length = min_segment_length
        self.split_cost = split_cost
        self.percent_pruning_margin = percent_pruning_margin
        self.middle_penalty_nudge = middle_penalty_nudge
        self.drop_pruning = drop_pruning

        # Storage for the CROPS results:
        self.change_points_metadata_ = None
        self.change_points_lookup_ = None

    def run_pelt(self, penalty: float) -> dict:
        """Run the CROPS algorithm for path solutions to penalized CPD.

        Parameters
        ----------
        X : np.ndarray
            Data to search for change points in.

        Returns
        -------
        cpd_results : dict
            Dictionary with penalty values as keys and tuples of change points and cost
            as values.
        """
        opt_cost, change_points = run_improved_pelt_array_based(
            self.cost,
            penalty=penalty,
            min_segment_length=self.min_segment_length,
            split_cost=self.split_cost,
            percent_pruning_margin=self.percent_pruning_margin,
            drop_pruning=self.drop_pruning,
        )

        return change_points

    def run_crops(self, X: np.ndarray):
        """Run the CROPS algorithm for path solutions to penalized CPD.

        Parameters
        ----------
        X : np.ndarray
            Data to search for change points in.
        """
        self.cost.fit(X)

        min_penalty_change_points = self.run_pelt(penalty=self.min_penalty)
        max_penalty_change_points = self.run_pelt(penalty=self.max_penalty)

        num_min_penalty_change_points = len(min_penalty_change_points)
        num_max_penalty_change_points = len(max_penalty_change_points)

        min_penalty_segmentation_cost = self.cost.evaluate_segmentation(
            min_penalty_change_points
        )
        max_penalty_segmentation_cost = self.cost.evaluate_segmentation(
            max_penalty_change_points
        )

        penalty_to_solution_dict: dict[float, (np.ndarray, float)] = dict()
        penalty_to_solution_dict[self.min_penalty] = (
            min_penalty_change_points,
            min_penalty_segmentation_cost,
        )
        penalty_to_solution_dict[self.max_penalty] = (
            max_penalty_change_points,
            max_penalty_segmentation_cost,
        )

        # Store intervals of penalty values in which to search for
        # differing numbers of change points.
        penalty_search_intervals = []
        if num_min_penalty_change_points > num_max_penalty_change_points + 1:
            # More than one change point in difference between min and max penalty.
            # Need to split the penalty intervals further.
            penalty_search_intervals.append((self.min_penalty, self.max_penalty))

        while len(penalty_search_intervals) > 0:
            # Pop the interval with the lowest penalty.
            low_penalty, high_penalty = penalty_search_intervals.pop(0)
            low_penalty_change_points, low_penalty_segmentation_cost = (
                penalty_to_solution_dict[low_penalty]
            )
            high_penalty_change_points, high_penalty_segmentation_cost = (
                penalty_to_solution_dict[high_penalty]
            )
            if len(low_penalty_change_points) > (len(high_penalty_change_points) + 1):
                # Need to compute the middle penalty value, to explore interval further:
                middle_penalty = (
                    (high_penalty_segmentation_cost - low_penalty_segmentation_cost)
                    / (len(low_penalty_change_points) - len(high_penalty_change_points))
                    * (
                        1.0 + self.middle_penalty_nudge
                    )  # Nudge the penalty to avoid numerical instability.
                )

                middle_penalty_change_points = self.run_pelt(penalty=middle_penalty)
                middle_penalty_segmentation_cost = self.cost.evaluate_segmentation(
                    middle_penalty_change_points
                )
                penalty_to_solution_dict[middle_penalty] = (
                    middle_penalty_change_points,
                    middle_penalty_segmentation_cost,
                )

                middle_penalty_matches_high_penalty = len(
                    middle_penalty_change_points
                ) == len(high_penalty_change_points)
                middle_penalty_matches_low_penalty = len(
                    middle_penalty_change_points
                ) == len(low_penalty_change_points)
                if middle_penalty_matches_high_penalty:
                    # The same number of change points for penalties in the interval
                    # [middle_penalty, high_penalty].
                    # Don't need to subdivide penalty intervals further.
                    continue
                elif middle_penalty_matches_low_penalty:
                    # raise ValueError("This should not happen!")
                    # In this case, PELT has failed to solve the penalized
                    # optimization problem exactly. Need to increase the penalty
                    # until we get at least one more change point than for
                    # low_penalty.

                    # We're in a case where due to numerical instability, we have
                    # the same number of change points for low_penalty and
                    # middle_penalty, although we should have the same number
                    # of change points for middle penalty and high_penalty.
                    # And for middle_penalty + epsilon, we would get the same
                    # number of change points as for high_penalty.

                    # For change detectors that only solve the penalised optimization
                    # problem approximately, we'll need to perform a linear search
                    # from the 'middle_penalty' penalty to the 'high_penalty' penalty.
                    # Until we fnd a penalty that gives us the same number of
                    # change points as for for high_penalty, or at least a penalty
                    # that gives fewer change points than for low_penalty.
                    # nudged_middle_penalty = middle_penalty * (1 + 1.0e-4)
                    # nudged_middle_penalty_change_points = self.run_pelt(
                    #     penalty=nudged_middle_penalty
                    # )
                    # nudged_middle_penalty_segmentation_cost = (
                    #     self.cost.evaluate_segmentation(
                    #         nudged_middle_penalty_change_points
                    #     )
                    # )
                    # penalty_to_solution_dict[nudged_middle_penalty] = (
                    #     nudged_middle_penalty_change_points,
                    #     nudged_middle_penalty_segmentation_cost,
                    # )

                    # Only add [middle_penalty, high_penalty] to the search intervals:
                    # penalty_search_intervals.append((middle_penalty, high_penalty))
                    # penalty_search_intervals.sort(key=lambda x: x[0])
                    ## Indicates that the 'improved_pelt_array_based' method has failed
                    ## to solve the penalized optimization problem exactly. Need to
                    ## increase the pruning margin... Not good...
                    raise ValueError(
                        "This should not happen! Number of change points should be"
                        " greater for the middle penalty than for the low penalty."
                    )
                    print("Warning: This should not happen!")
                    self.percent_pruning_margin = max(
                        2.0 * self.percent_pruning_margin, 0.1
                    )
                    print(
                        "Setting percent_pruning_margin to: ",
                        self.percent_pruning_margin,
                    )
                    return self.run_crops(X=X)
                else:
                    # Number of change points for middle penalty is different from both
                    # low_penalty and high_penalty. Need to explore the penalty
                    # intervals further.
                    # Explore penalties above and below the middle penalty:
                    penalty_search_intervals.append((low_penalty, middle_penalty))
                    penalty_search_intervals.append((middle_penalty, high_penalty))

                    # Sort the intervals by lower penalty:
                    penalty_search_intervals.sort(key=lambda x: x[0])

        # # Convert the dict lookup results to a list of tuples:
        # list_cpd_results = [
        #     (len(change_points), penalty, change_points, segmentation_cost)
        #     for (penalty, (change_points, segmentation_cost)) in cpd_results.items()
        # ]
        # # When different penalties give the same number of change points, we want to
        # # keep the segmentation with the lowest penalty. So we sort by number of
        # # change points, and then negative penalty, in reverse order.
        # list_cpd_results.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        # # Remove duplicates, and keep the first one:
        # encountered_num_change_points = set()
        # unique_cpd_results = []
        # for i in range(len(list_cpd_results)):
        #     num_change_points = list_cpd_results[i][0]
        #     if num_change_points in encountered_num_change_points:
        #         continue
        #     else:
        #         unique_cpd_results.append(list_cpd_results[i])
        #         encountered_num_change_points.add(num_change_points)
        self.store_crops(penalty_to_solution_dict)
        return self.change_points_metadata_

    def store_crops(self, penalty_to_solution_dict: list) -> None:
        """Store the CROPS results in the change point detector.

        Parameters
        ----------
        cpd_results : list
            List of tuples with:
            - (number of change points, penalty, change points, segmentation cost).

        """
        metadata_df, change_points_dict = format_crops_results(
            penalty_to_solution_dict=penalty_to_solution_dict
        )
        self.change_points_metadata_ = metadata_df
        self.change_points_lookup_ = change_points_dict


class GenericCROPS(BaseChangeDetector):
    """CROPS algorithm for path solutions to penalized CPD.

    Reference: https://arxiv.org/pdf/1412.3617

    Parameters
    ----------
    cost : BaseCost
        The cost function to use.
    min_penalty : float
        The minimum penalty to use.
    max_penalty : float
        The maximum penalty to use.
    change_detector : BaseChangeDetector
        The change point detector to use. For theoretical guarantees,
        the change point detector should be a PELT-kind of detector,
        which solves a linearly penalized optimal segmentation problem.
        If None, will use the PELT detector. Default is None.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": ["johannvk"],
        "fit_is_empty": True,
    }

    def __init__(
        self,
        change_detector: BaseChangeDetector,
        min_penalty: float,
        max_penalty: float,
        segmentation_cost: BaseCost,
    ):
        super().__init__()
        self.change_detector = change_detector
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.segmentation_cost: BaseCost = segmentation_cost

    def run_change_point_prediction(self, penalty: float, X: np.ndarray) -> dict:
        """Run the CROPS algorithm for path solutions to penalized CPD.

        Parameters
        ----------
        X : np.ndarray
            Data to search for change points in.

        Returns
        -------
        cpd_results : dict
            Dictionary with penalty values as keys and tuples of change points and cost
            as values.
        """
        # change_point_detector = self.change_detector(cost=self.cost, penalty=penalty)
        self.change_detector.update_penalty(penalty=penalty)
        self.change_detector.fit(X)  # May be empty, but need to call fit at least once?
        change_points = self.change_detector.predict(X)

        return change_points

    def _predict(self, X: np.ndarray):
        """Run the CROPS algorithm for path solutions to penalized CPD.

        Parameters
        ----------
        X : np.ndarray
            Data to search for change points in.
        """
        cpd_results: dict[float, (np.ndarray, float)] = dict()
        self.segmentation_cost.fit(X)

        min_penalty_change_points = self.run_change_point_prediction(
            penalty=self.min_penalty, X=X
        )
        num_min_penalty_change_points = len(min_penalty_change_points)

        max_penalty_change_points = self.run_change_point_prediction(
            penalty=self.max_penalty, X=X
        )
        num_max_penalty_change_points = len(max_penalty_change_points)

        # Annoying to have to do this, but need to dig into the PELT object to get the
        # segmentation cost.
        # min_penalty_segmentation_cost = (
        #     min_penalty_cpd._penalised_cost.scorer_.evaluate_interval(
        #         min_penalty_change_points["ilocs"]
        #     )
        # )
        min_penalty_segmentation_cost = self.segmentation_cost.evaluate_segmentation(
            min_penalty_change_points["ilocs"]
        )
        max_penalty_segmentation_cost = self.segmentation_cost.evaluate_segmentation(
            max_penalty_change_points["ilocs"]
        )

        cpd_results[self.min_penalty] = (
            min_penalty_change_points,
            min_penalty_segmentation_cost,
        )
        cpd_results[self.max_penalty] = (
            max_penalty_change_points,
            max_penalty_segmentation_cost,
        )

        if num_min_penalty_change_points <= num_max_penalty_change_points + 1:
            # Less than two change points in difference between min and max penalty.
            # No need to split the penalty intervals further.
            return cpd_results

        # Calculate the middle penalty value, to explore penalty interval further:
        first_middle_penalty = (
            max_penalty_segmentation_cost - min_penalty_segmentation_cost
        ) / (num_min_penalty_change_points - num_max_penalty_change_points)

        first_middle_penalty_change_points = self.run_change_point_prediction(
            penalty=first_middle_penalty, X=X
        )
        num_first_middle_penalty_change_points = len(first_middle_penalty_change_points)
        first_middle_penalty_segmentation_cost = (
            self.segmentation_cost.evaluate_segmentation(
                first_middle_penalty_change_points["ilocs"]
            )
        )
        cpd_results[first_middle_penalty] = (
            first_middle_penalty_change_points,
            first_middle_penalty_segmentation_cost,
        )

        if num_first_middle_penalty_change_points == num_max_penalty_change_points:
            # No need to split the penalty intervals further.
            return cpd_results

        # Want a Heap of some kind, so that we can pop and push intervals onto it.
        penalty_intervals = [
            (self.min_penalty, first_middle_penalty),
            (first_middle_penalty, self.max_penalty),
        ]

        while len(penalty_intervals) > 0:
            # Pop the interval with the lowest penalty.
            low_penalty, high_penalty = penalty_intervals.pop(0)
            low_penalty_change_points, low_penalty_segmentation_cost = cpd_results[
                low_penalty
            ]
            high_penalty_change_points, high_penalty_segmentation_cost = cpd_results[
                high_penalty
            ]
            if len(low_penalty_change_points) > (len(high_penalty_change_points) + 1):
                # Need to compute the middle penalty value, to explore interval further:
                middle_penalty = (
                    (high_penalty_segmentation_cost - low_penalty_segmentation_cost)
                    / (len(low_penalty_change_points) - len(high_penalty_change_points))
                    * (
                        1.0 + 1.0e-4
                    )  # Nudge the penalty to avoid numerical instability.
                )

                middle_penalty_change_points = self.run_change_point_prediction(
                    penalty=middle_penalty, X=X
                )
                middle_penalty_segmentation_cost = (
                    self.segmentation_cost.evaluate_segmentation(
                        middle_penalty_change_points["ilocs"]
                    )
                )
                middle_penalty_matches_high_penalty = len(
                    middle_penalty_change_points
                ) == len(high_penalty_change_points)
                middle_penalty_matches_low_penalty = len(
                    middle_penalty_change_points
                ) == len(low_penalty_change_points)
                if middle_penalty_matches_high_penalty:
                    # The same number of change points for penalties in the interval
                    # [middle_penalty, high_penalty], or [low_penalty, middle_penalty].
                    # Don't need to explore the penalty # intervals further.
                    cpd_results[middle_penalty] = (
                        middle_penalty_change_points,
                        middle_penalty_segmentation_cost,
                    )
                    continue
                elif middle_penalty_matches_low_penalty:
                    # We're in a case where due to numerical instability, we have
                    # the same number of change points for low_penalty and
                    # middle_penalty, although we should have the same number
                    # of change points for middle penalty and high_penalty.
                    # And for middle_penalty + epsilon, we would get the same
                    # number of change points as for high_penalty.

                    # For change detectors that only solve the penalised optimization
                    # problem approximately, we'll need to perform a linear search
                    # from the 'middle_penalty' penalty to the 'high_penalty' penalty.
                    # Until we fnd a penalty that gives us the same number of
                    # change points as for for high_penalty, or at least a penalty
                    # that gives fewer change points than for low_penalty.
                    nudged_middle_penalty = middle_penalty * (1 + 1.0e-4)
                    nudged_middle_penalty_change_points = (
                        self.run_change_point_prediction(
                            penalty=nudged_middle_penalty, X=X
                        )
                    )
                    nudged_middle_penalty_segmentation_cost = (
                        self.segmentation_cost.evaluate_segmentation(
                            nudged_middle_penalty_change_points["ilocs"]
                        )
                    )

                    cpd_results[nudged_middle_penalty] = (
                        nudged_middle_penalty_change_points,
                        nudged_middle_penalty_segmentation_cost,
                    )
                else:
                    # Explore penalties above and below the middle penalty:
                    penalty_intervals.append((low_penalty, middle_penalty))
                    penalty_intervals.append((middle_penalty, high_penalty))

                    # Sort the intervals by lower penalty:
                    penalty_intervals.sort(key=lambda x: x[0])

                    cpd_results[middle_penalty] = (
                        middle_penalty_change_points,
                        middle_penalty_segmentation_cost,
                    )

        return cpd_results
