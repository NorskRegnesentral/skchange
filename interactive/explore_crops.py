"""
Exploration of CROPS (Changepoints for a Range Of PenaltieS) algorithm.

This module explores the implementation and behavior of the CROPS algorithm
with various cost functions, focusing on comparison between different pruning
strategies and performance evaluation.
"""

from functools import reduce

import numpy as np
import ruptures as rpt
from ruptures.base import BaseCost as rpt_BaseCost

from skchange.change_detectors import PELT, SeededBinarySegmentation
from skchange.change_detectors._crops import CROPS_PELT, GenericCROPS
from skchange.change_detectors._pelt import (
    run_improved_pelt_array_based,
    run_restricted_optimal_partitioning,
)
from skchange.change_scores._from_cost import to_change_score
from skchange.costs import GaussianCost, L1Cost, L2Cost
from skchange.datasets import generate_alternating_data

# %% Testin out the "automatic" elbow detection method:
# cost = L2Cost()
cost = GaussianCost()
# # # cost = L1Cost()
min_penalty = 1.0e-2
max_penalty = 5.0e3

# # Fails with 'min_segment_length=10' and 'percent_pruning_margin=0.0'
# # Perhaps due to numerical issues, it's a bit unclear.
min_segment_length = 10
percent_pruning_margin = 0.0

# Generate test data:
dataset = generate_alternating_data(
    n_segments=5,
    segment_length=100,
    p=1,
    mean=3.0,
    variance=4.0,
    random_state=42,
)

fitted_cost = cost.fit(dataset.values)


# %%
class RupturesGaussianCost(rpt_BaseCost):
    """Custom cost for Gaussian (mean-var) cost."""

    # The 2 following attributes must be specified for compatibility.
    model = ""
    min_size = 2

    def fit(self, signal) -> "RupturesGaussianCost":
        """Fit the cost function to the data."""
        self.signal = signal
        self.cost = GaussianCost().fit(signal)
        return self

    def error(self, start: int, end: int) -> np.ndarray:
        """Compute the cost of a segment."""
        cuts = np.array([start, end]).reshape(-1, 2)
        return self.cost.evaluate(cuts)


# %%
# Run Ruptures with 'jump > 1'.
jump_step = 10
jump_penalty = 104.362860 / 1.01
rpt_PELT = rpt.Pelt(
    custom_cost=RupturesGaussianCost(),
    min_size=jump_step,
    jump=jump_step,
)
rpt_PELT.fit(dataset.values)
jump_rpt_pelt_cpts = np.array(rpt_PELT.predict(pen=jump_penalty)[:-1])

# Compared to 'JumpCost' from Skchange:
# from skchange.change_detectors._crops import JumpCost

# jump_cost = JumpCost(
#     cost=cost,
#     jump=jump_step,
# )
# jump_cost.fit(dataset.values)

# pelt_opt_cost, jump_PELT_change_points = run_improved_pelt_array_based(
#     cost=jump_cost, penalty=jump_penalty, min_segment_length=1
# )
# # Rescale the change points to the original data:
# jump_PELT_change_points = jump_PELT_change_points * jump_step

# # With end-point change points:
# with_end_jump_PELT_change_points = np.concatenate(
#     [jump_PELT_change_points, [len(dataset.values)]]
# )
with_end_jump_rpt_pelt_cpts = np.concatenate(
    [jump_rpt_pelt_cpts, [len(dataset.values)]]
)
# assert np.diff(with_end_jump_PELT_change_points).min() >= jump_step
assert np.diff(with_end_jump_rpt_pelt_cpts).min() >= jump_step
# assert np.array_equal(jump_PELT_change_points, jump_rpt_pelt_cpts), (
#     "JumpCost and Ruptures PELT do not match!"
# )

# %%
# change_point_detector._cost.fit(dataset.values)

# To refine the 'JUMP' changepoints, can we run 'Optimal partitioning'
# on a restricted set of 'start' points, around the 'JUMP' change points?
# I.e. [change_point - min_segment_length, change_point + min_segment_length]
# for each 'JUMP' change point?
change_point_detector = CROPS_PELT(
    cost=cost,
    min_penalty=min_penalty,
    max_penalty=max_penalty,
    min_segment_length=min_segment_length,
    # min_segment_length=cost.min_size,
    # percent_pruning_margin=percent_pruning_margin,
    drop_pruning=False,
)

# Fit the change point detector:
change_point_detector.fit(dataset)

direct_results = change_point_detector.run_crops(dataset.values)
direct_results["optimal_value"] = direct_results["segmentation_cost"] + direct_results[
    "penalty"
] * (direct_results["num_change_points"] + 1)
# %%
# %timeit margin_results = change_point_detector.run_crops(dataset.values)

# %%
no_prune_cpd = CROPS_PELT(
    cost=cost,
    min_penalty=min_penalty,
    max_penalty=max_penalty,
    min_segment_length=min_segment_length,
    # min_segment_length=cost.min_size,
    drop_pruning=True,
)
no_prune_cpd.fit(dataset)
no_prune_results = no_prune_cpd.run_crops(dataset.values)
no_prune_results["optimal_value"] = no_prune_results[
    "segmentation_cost"
] + no_prune_results["penalty"] * (no_prune_results["num_change_points"] + 1)

# %%
# %timeit no_prune_results = no_prune_cpd.run_crops(dataset.values)
# %%
num_cpts = 10
exact_cpts = no_prune_cpd.change_points_lookup_[num_cpts]
jump_cpts = change_point_detector.change_points_lookup_[num_cpts]
jump_penalty = direct_results[direct_results["num_change_points"] == num_cpts][
    "penalty"
].values[0]

admissable_starts = reduce(
    lambda x, y: x | y,
    [
        set(range(cpt - (min_segment_length - 1), cpt + (min_segment_length - 1) + 1))
        for cpt in jump_cpts
    ],
)
refined_pelt_cost, refined_cpts = run_restricted_optimal_partitioning(
    cost=cost,
    penalty=jump_penalty,
    min_segment_length=min_segment_length,
    admissable_starts=admissable_starts,
)

# %%
# Regular PELT min_segment_length = 10, penalty = 5.892686
test_penalty = 5.892686
regular_pelt_cpts = (
    PELT(
        cost=cost,
        penalty=test_penalty,
        min_segment_length=min_segment_length,
    )
    .fit(dataset)
    .predict(dataset)
)["ilocs"].to_numpy()
(
    cost.evaluate_segmentation(regular_pelt_cpts)
    + (len(regular_pelt_cpts) - 1) * test_penalty
)

no_prune_regular_pelt_cpts = (
    PELT(
        cost=cost,
        penalty=test_penalty,
        min_segment_length=min_segment_length,
        drop_pruning=True,
    )
    .fit(dataset)
    .predict(dataset)
)["ilocs"].to_numpy()
(
    cost.evaluate_segmentation(no_prune_regular_pelt_cpts)
    + (len(no_prune_regular_pelt_cpts) - 1) * test_penalty
)

# %%
# Check that the results are as expected:
if len(direct_results) != len(no_prune_results):
    print(
        f"Length of margin results ({len(direct_results)}) does not match"
        f" length of no prune results ({len(no_prune_results)})"
    )
else:
    print("Len(margin_results):", len(direct_results))
    print((direct_results == no_prune_results).all())

# %%
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # Add a row for 'zero' change points:
# if results["num_change_points"].min() != 0:
#     with_no_change_points_df = pd.concat(
#         [
#             results,
#             pd.DataFrame(
#                 {
#                     "num_change_points": [0],
#                     "segmentation_cost": [
#                         cost.fit(dataset.values).evaluate_segmentation(np.array([]))
#                     ],
#                     "penalty": [np.inf],
#                 }
#             ),
#         ],
#         ignore_index=True,
#     )
# else:
#     with_no_change_points_df = results.copy()

# # Plot segmentation cost vs. number of change points:
# plt.plot(
#     with_no_change_points_df["num_change_points"],
#     with_no_change_points_df["segmentation_cost"],
#     label="Segmentation Cost",
# )

# with_no_change_points_df["V_k"] = (
#     with_no_change_points_df["segmentation_cost"]
#     - with_no_change_points_df["segmentation_cost"].min()
# )

# with_no_change_points_df["elbow_loss"] = (
#     with_no_change_points_df["V_k"]
#     + (
#         with_no_change_points_df["V_k"].max()
#         / with_no_change_points_df["num_change_points"].max()
#     )
#     * with_no_change_points_df["num_change_points"]
# )

# with_no_change_points_df["step_percent_decrease"] = (
#     -100.0
#     * (with_no_change_points_df["segmentation_cost"][::-1].diff()[::-1])
#     / (with_no_change_points_df["segmentation_cost"].shift(-1))
# )

# with_no_change_points_df["percent_decrease_from_top"] = (
#     -100.0
#     * (with_no_change_points_df["segmentation_cost"][::-1].diff()[::-1])
#     / (with_no_change_points_df["segmentation_cost"].max())
# )

# argmin_elbow_loss = with_no_change_points_df["elbow_loss"].argmin()
# with_no_change_points_df.loc[argmin_elbow_loss, :]

# with_no_change_points_df["bic_loss"] = with_no_change_points_df[
#     "segmentation_cost"
# ] + cost.get_param_size(1) * with_no_change_points_df["num_change_points"] * np.log(
#     len(dataset)
# )
# argmin_bic_loss = with_no_change_points_df["bic_loss"].argmin()
# with_no_change_points_df.loc[argmin_bic_loss, :]


# min_percent_decrease = 1.0
# restricted_to_min_n_percent_decrease_from_top_df = pd.concat(
#     [
#         with_no_change_points_df[
#             with_no_change_points_df["percent_decrease_from_top"] > min_percent_decrease
#         ]
#         .copy()
#         .reset_index(drop=True),
#         pd.DataFrame(
#             {
#                 "num_change_points": [0],
#                 "segmentation_cost": [
#                     cost.fit(dataset.values).evaluate_segmentation(np.array([]))
#                 ],
#                 "penalty": [np.inf],
#             }
#         ),
#     ],
#     ignore_index=True,
# )

# restricted_to_min_n_percent_decrease_from_top_df["V_k"] = (
#     restricted_to_min_n_percent_decrease_from_top_df["segmentation_cost"]
#     - restricted_to_min_n_percent_decrease_from_top_df["segmentation_cost"].min()
# )

# # Rerun the elbow loss calculation:
# restricted_to_min_n_percent_decrease_from_top_df["elbow_loss"] = (
#     restricted_to_min_n_percent_decrease_from_top_df["V_k"]
#     + (
#         restricted_to_min_n_percent_decrease_from_top_df["V_k"].max()
#         / restricted_to_min_n_percent_decrease_from_top_df["num_change_points"].max()
#     )
#     * restricted_to_min_n_percent_decrease_from_top_df["num_change_points"]
# )
# restricted_argmin_elbow_loss = restricted_to_min_n_percent_decrease_from_top_df[
#     "elbow_loss"
# ].argmin()
# restricted_to_min_n_percent_decrease_from_top_df.loc[restricted_argmin_elbow_loss, :]
