from skchange.change_detectors import PELT, SeededBinarySegmentation
from skchange.change_detectors._crops import CROPS_PELT, GenericCROPS
from skchange.change_scores._from_cost import to_change_score
from skchange.costs import GaussianCost, L1Cost, L2Cost
from skchange.datasets import generate_alternating_data


def test_pelt_crops():
    """Test the CROPS algorithm for path solutions to penalized CPD.

    Reference: https://arxiv.org/pdf/1412.3617
    """
    cost = L2Cost()
    min_penalty = 0.5
    max_penalty = 50.0

    change_point_detector = CROPS_PELT(
        cost=cost,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
    )

    # Generate test data:
    dataset = generate_alternating_data(
        n_segments=2,
        segment_length=100,
        p=1,
        mean=3.0,
        variance=4.0,
        random_state=42,
    )

    # Fit the change point detector:
    change_point_detector.fit(dataset)
    results = change_point_detector.run_crops(dataset.values)
    # Check that the results are as expected:
    assert len(results) == 64


def test_generic_crops_on_SeededBinarySegmentation():
    cost = GaussianCost()
    min_penalty = 0.1
    max_penalty = 100.0
    change_detector = SeededBinarySegmentation(change_score=to_change_score(cost))
    # Initialize penalization interval change point detector:
    crops_change_detector = GenericCROPS(
        change_detector=change_detector,
        segmentation_cost=cost,
        min_penalty=min_penalty,
        max_penalty=max_penalty,
    )

    # Generate test data:
    dataset = generate_alternating_data(
        n_segments=2,
        segment_length=100,
        p=1,
        mean=3.0,
        variance=4.0,
        random_state=42,
    )

    # Fit the change point detector:
    crops_change_detector.fit(dataset)
    results = crops_change_detector.predict(dataset.values)
    # Check that the results are as expected:
    assert results is not None


# %% Testin out the "automatic" elbow detection method:
# cost = L2Cost()
# cost = GaussianCost()
# # cost = L1Cost()
# min_penalty = 5.0e-1
# max_penalty = 5.0e3

# change_point_detector = CROPS_PELT(
#     cost=cost,
#     min_penalty=min_penalty,
#     max_penalty=max_penalty,
#     min_segment_length=10,
#     # min_segment_length=cost.min_size,
#     percent_pruning_margin=5.0,
# )

# # Generate test data:
# dataset = generate_alternating_data(
#     n_segments=5,
#     segment_length=100,
#     p=1,
#     mean=3.0,
#     variance=4.0,
#     random_state=42,
# )

# # Fit the change point detector:
# change_point_detector.fit(dataset)
# results = change_point_detector.run_crops(dataset.values)

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
