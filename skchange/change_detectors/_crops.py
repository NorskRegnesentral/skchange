"""Implementation of the CROPS algorithm for path solutions to penalized CPD."""

import warnings
from functools import reduce

import numpy as np
import pandas as pd

from ..change_detectors._pelt import (
    run_improved_pelt_array_based,
    run_pelt_with_jump,
    run_restricted_optimal_partitioning,
)
from ..change_scores import ContinuousLinearTrendScore
from ..costs.base import BaseCost
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
    # Reverse the order of the list, so that the lowest number
    # of change points comes first.
    change_points_metadata = [x[0:3] for x in unique_cpd_results][::-1]
    penalty_change_point_metadata_df = pd.DataFrame(
        change_points_metadata,
        columns=[
            "num_change_points",
            "penalty",
            "segmentation_cost",
        ],
    )
    penalty_change_point_metadata_df["num_change_points"] = (
        penalty_change_point_metadata_df["num_change_points"].astype(int)
    )

    penalty_change_point_metadata_df["optimum_value"] = (
        penalty_change_point_metadata_df["segmentation_cost"]
        + penalty_change_point_metadata_df["penalty"]
        * (penalty_change_point_metadata_df["num_change_points"] + 1)
    )
    return penalty_change_point_metadata_df, change_points_dict


def segmentation_bic_value(cost: BaseCost, change_points: np.ndarray) -> float:
    """Calculate the BIC score for a given segmentation.

    Parameters
    ----------
    cost : BaseCost
        The cost function to use.
    change_points : np.ndarray
        The change points to use.

    Returns
    -------
    float
        The BIC score for the given segmentation.
    """
    cost.check_is_fitted()
    num_segments = len(change_points) + 1
    data_dim = cost._X.shape[1]
    num_parameters = cost.get_model_size(data_dim)
    n_samples = cost.n_samples()
    return cost.evaluate_segmentation(
        change_points
    ) + num_parameters * num_segments * np.log(n_samples)


def crops_elbow_score(
    num_change_points_and_optimum_value_df: pd.DataFrame,
) -> pd.Series:
    """Calculate the elbow cost for a given segmentation.

    Specifically, the elbow score is calculated as the evidence for a change
    of slope in the segmentation cost as a function of the number of
    change points, at each intermediate number of change points.
    We cannot calculate the elbow score for the first and last number of
    change points, as there are not enough segmentations to calculate a change
    in slope before or after the first and last number of change points.

    Parameters
    ----------
    num_change_points : int
        The number of change points.
    segmentation_cost : float
        The segmentation cost.

    Returns
    -------
    pd.Series
        The elbow cost for each number of change points.
    """
    num_segmentations = len(num_change_points_and_optimum_value_df)
    if num_segmentations < 3:
        # Not enough segmentations to calculate the elbow cost.
        warnings.warn(
            f"Not enough segmentations {num_segmentations} to calculate "
            "the elbow cost. Returning -np.inf for all segmentations."
        )
        return pd.Series([-np.inf] * num_segmentations)

    # Calculate the elbow (change in slope) cost for each number of change points:
    continuous_linear_trend_score = ContinuousLinearTrendScore(
        time_column="num_change_points"
    )
    continuous_linear_trend_score.fit(num_change_points_and_optimum_value_df)

    # Construct cuts at all intermediate number of change points.
    # I.e. not at the first and last number of change points.
    score_eval_cuts = np.column_stack(
        (
            np.repeat(0, num_segmentations - 2),
            np.arange(1, num_segmentations - 1),
            np.repeat(num_segmentations, num_segmentations - 2),
        )
    )

    # Pad the elbow score with `-np.inf` for the first and last segmentations,
    # which we cannot calculate a "change in slope" score for.
    elbow_score = np.concatenate(
        (
            np.array([-np.inf]),
            continuous_linear_trend_score.evaluate(score_eval_cuts).reshape(-1),
            np.array([-np.inf]),
        )
    )
    return elbow_score


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
    segmentation_selection : str, default="bic"
        The selection criterion to use for selecting the
        best segmentation among the optimal segmentations for
        the penalty range `[min_penalty, max_penalty]`.
        Options: ["bic", "elbow"]. Default is "bic".
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
        "fit_is_empty": False,
    }

    def __init__(
        self,
        cost: BaseCost,
        min_penalty: float,
        max_penalty: float,
        segmentation_selection: str = "bic",
        min_segment_length: int = 1,
        split_cost: float = 0.0,
        percent_pruning_margin: float = 0.0,
        middle_penalty_nudge: float = 1.0e-4,
        drop_pruning: bool = False,
    ):
        super().__init__()
        self.cost: BaseCost = cost
        self._cost: BaseCost = cost.clone()

        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.segmentation_selection = segmentation_selection
        self.min_segment_length = min_segment_length
        self.split_cost = split_cost
        self.percent_pruning_margin = percent_pruning_margin
        self.middle_penalty_nudge = middle_penalty_nudge
        self.drop_pruning = drop_pruning

        if segmentation_selection not in ["bic", "elbow"]:
            raise ValueError(
                f"Invalid selection criterion: {segmentation_selection}. "
                "Must be one of ['bic', 'elbow']."
            )

        # Storage for the CROPS results:
        self.change_points_metadata_: pd.DataFrame | None = None
        self.change_points_lookup_: dict[int, np.ndarray] | None = None

    def _fit(self, X, y=None):
        """Fit the cost.

        Parameters
        ----------
        X : np.ndarray
            Data to search for change points in.
        """
        self._cost.fit(X, y)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Run the CROPS algorithm for path solutions to penalized CPD.

        Parameters
        ----------
        X : np.ndarray
            Data to search for change points in.

        Returns
        -------
        np.ndarray
            The change points for the given number of change points.
        """
        self.run_crops(X=X)

        if self.segmentation_selection == "elbow":
            # Select the best segmentation using the elbow criterion.
            change_in_slope_df = self.change_points_metadata_[
                ["num_change_points", "optimum_value"]
            ].copy()
            # Subrtract the minimum value from the optimum value
            # to improve conditioning number of linreg problems.
            change_in_slope_df["optimum_value"] = (
                change_in_slope_df["optimum_value"]
                - change_in_slope_df["optimum_value"].min()
            )
            self.change_points_metadata_["elbow_score"] = crops_elbow_score(
                change_in_slope_df,
            )
            best_num_change_points = self.change_points_metadata_.sort_values(
                by="elbow_score", ascending=False
            )["num_change_points"].iloc[0]

        elif self.segmentation_selection == "bic":
            # Select the best segmentation using the BIC criterion.
            self.change_points_metadata_["bic_value"] = self.change_points_metadata_[
                "num_change_points"
            ].apply(
                lambda num_change_points: segmentation_bic_value(
                    cost=self._cost,
                    change_points=self.change_points_lookup_[num_change_points],
                )
            )
            best_num_change_points = self.change_points_metadata_.sort_values(
                by="bic_value", ascending=True
            )["num_change_points"].iloc[0]

        else:
            raise ValueError(  # pragma: no cover
                f"Invalid selection criterion: {self.segmentation_selection}. "
                "Must be one of ['bic', 'elbow']."
            )

        return self.change_points_lookup_[best_num_change_points]

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
        if self.min_segment_length > 1 and not self.drop_pruning:
            opt_cost, change_points = run_pelt_with_jump(
                self._cost,
                penalty=penalty,
                jump_step=self.min_segment_length,
                split_cost=self.split_cost,
            )
        else:
            opt_cost, change_points = run_improved_pelt_array_based(
                self._cost,
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
        self._cost.fit(X)

        min_penalty_change_points = self.run_pelt(penalty=self.min_penalty)
        max_penalty_change_points = self.run_pelt(penalty=self.max_penalty)

        num_min_penalty_change_points = len(min_penalty_change_points)
        num_max_penalty_change_points = len(max_penalty_change_points)

        min_penalty_segmentation_cost = self._cost.evaluate_segmentation(
            min_penalty_change_points
        )
        max_penalty_segmentation_cost = self._cost.evaluate_segmentation(
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
                middle_penalty_segmentation_cost = self._cost.evaluate_segmentation(
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
                    # The same number of change points for penalties in
                    # the interval [middle_penalty, high_penalty].
                    # Don't need to subdivide penalty intervals further.
                    continue
                elif middle_penalty_matches_low_penalty:
                    raise ValueError(  # pragma: no cover
                        "PELT optimization has not been solved exactly! "
                        "Number of change points should be greater for the "
                        "middle penalty than for the low penalty. "
                        "Attempt to set the `split_cost` parameter to a "
                        "non-zero value, or increase `percenct_pruning_margin`."
                    )
                else:
                    # Number of change points for middle penalty is different from both
                    # low_penalty and high_penalty. Need to explore further.
                    penalty_search_intervals.append((low_penalty, middle_penalty))
                    penalty_search_intervals.append((middle_penalty, high_penalty))

                    # Sort the intervals by lower penalty:
                    penalty_search_intervals.sort(key=lambda x: x[0])

        self.store_path_solution(penalty_to_solution_dict)
        return self.change_points_metadata_

    def store_path_solution(self, penalty_to_solution_dict: list) -> None:
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

    def retrieve_change_points(
        self, num_change_points: int, refine_change_points: bool = True
    ) -> np.ndarray:
        """Retrieve the change points for a given number of change points.

        Parameters
        ----------
        num_change_points : int
            The number of change points to retrieve.
        refine_change_points : bool, default=True
            Whether to refine the change points using the restricted
            optimal partitioning algorithm. This may change the number
            of change points detected.
            If False, the detected change points will be returned as is,
            with potential change points only every `min_segment_length`.

        Returns
        -------
        np.ndarray
            The change points for the given number of change points.
        """
        if self.change_points_lookup_ is None:
            raise ValueError("CROPS results have not been computed yet.")

        if num_change_points not in self.change_points_lookup_:
            raise ValueError(
                f"Number of change points {num_change_points}"
                " not found in CROPS results."
            )

        if (
            not refine_change_points
            or self.min_segment_length == 1
            or self.drop_pruning
        ):
            return self.change_points_lookup_[num_change_points]
        else:
            # Refine the change points using restricted OptPart algorithm:
            orig_change_points = self.change_points_lookup_[num_change_points]
            penalty = self.change_points_metadata_.loc[
                self.change_points_metadata_["num_change_points"] == num_change_points,
                "penalty",
            ].values[0]

            admissable_starts = reduce(
                lambda x, y: x | y,
                [
                    set(
                        range(
                            cpt - (self.min_segment_length - 1),
                            cpt + (self.min_segment_length - 1) + 1,
                        )
                    )
                    for cpt in orig_change_points
                ],
            )

            refined_costs, refined_change_points = run_restricted_optimal_partitioning(
                cost=self._cost,
                penalty=penalty,
                min_segment_length=self.min_segment_length,
                admissable_cpts=admissable_starts,
            )

            return refined_change_points

    @classmethod
    def get_test_params(cls, parameter_set: str = "default") -> dict:
        """Get test parameters for the CROPS algorithm."""
        from skchange.costs import L2Cost

        return [
            {
                "cost": L2Cost(),
                "min_penalty": 0.5,
                "max_penalty": 50.0,
                "segmentation_selection": "bic",
                "min_segment_length": 5,
                "split_cost": 0.0,
                "percent_pruning_margin": 0.0,
                "middle_penalty_nudge": 1.0e-4,
                "drop_pruning": False,
            },
            {
                "cost": L2Cost(),
                "min_penalty": 1.0,
                "max_penalty": 10.0,
                "segmentation_selection": "elbow",
                "min_segment_length": 2,
                "split_cost": 0.0,
                "percent_pruning_margin": 0.0,
                "middle_penalty_nudge": 1.0e-4,
                "drop_pruning": True,
            },
        ]
