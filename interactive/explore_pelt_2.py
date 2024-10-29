import numpy as np
import pandas as pd
import plotly.express as px

from skchange.change_detectors.pelt import (
    pelt_partition_cost,
    run_optimal_partitioning,
    run_pelt,
)
from skchange.change_detectors.tests.test_pelt import run_pelt_old
from skchange.costs.cost_factory import cost_factory
from skchange.datasets.generate import generate_alternating_data

# %%
# X = np.random.randn(100, 1)
# X = generate_alternating_data(
#     n_segments=5, segment_length=20, p=1, random_state=5, mean=10.5, variance=0.5
# )[0].values.reshape(-1, 1)

X_complex_10_segments_n_200 = generate_alternating_data(
    n_segments=10, segment_length=20, p=1, random_state=10, mean=10.5, variance=0.5
)[0].values.reshape(-1, 1)
# X_complex_10_segments_n_200 = generate_alternating_data(
#     n_segments=10, segment_length=20, p=1, random_state=5, mean=10.5, variance=0.5
# )[0].values.reshape(-1, 1)

# Write the data to a file for testing with R:
# pd.Series(X_complex_10_segments_n_200.flatten()).to_csv(
#     "segments_10_n_200_random_state_10_data.csv", index=False, header=False
# )

# # cost_func, cost_init_func = cost_factory("mean")
cost_func, cost_init_func = cost_factory("mean")

def min_segment_cost_func(cost_function, min_segment_length):
    def limited_cost_func(params, starts, ends):
        cost = cost_function(params, starts, ends)
        return np.where(ends - starts + 1 < min_segment_length, np.inf, cost)

    return limited_cost_func


def compute_segment_lengths(change_points, num_obs):
    """
    Calculate the lengths of segments between change points.

    Parameters
    ----------
    change_points (list of int): Indices of the change points.
    num_obs (int): Total number of observations.

    Returns
    -------
    numpy.ndarray
        Array of segment lengths.
    """
    segment_lengths = []
    segment_start = 0
    for change_point in change_points:
        segment_length = change_point - segment_start + 1
        segment_lengths.append(segment_length)
        segment_start = change_point + 1
    last_segment_length = num_obs - segment_start

    return np.array(segment_lengths + [last_segment_length])


# %%
from pelt import Pelt

cpd = Pelt(
    cost="mean",
    penalty_scale=100.0 / (2 * 1 * np.log(len(X_complex_10_segments_n_200))),
    min_segment_length=21,
)
cpd.fit_transform(X_complex_10_segments_n_200)

sparse_change_points: pd.Series = cpd.predict(X_complex_10_segments_n_200)
dense_change_points = cpd.sparse_to_dense(
    sparse_change_points, range(len(X_complex_10_segments_n_200))
)
pelt_segment_lengths = compute_segment_lengths(
    sparse_change_points.to_list(), len(X_complex_10_segments_n_200)
)


# %%
pelt_penalty = 100.0
min_segment_length = 30

old_opt_costs, old_changepoints = run_pelt_old(
    X_complex_10_segments_n_200,
    cost_func,
    cost_init_func,
    penalty=pelt_penalty,
    min_segment_length=min_segment_length,
)
compute_segment_lengths(old_changepoints, len(X_complex_10_segments_n_200))

new_opt_costs, new_changepoints = run_pelt(
    X_complex_10_segments_n_200,
    cost_func,
    # min_segment_cost_func(cost_function=cost_func, min_segment_length=1),
    cost_init_func,
    penalty=pelt_penalty,
    min_segment_length=min_segment_length,
)
compute_segment_lengths(new_changepoints, len(X_complex_10_segments_n_200))

opt_part_costs, opt_part_changepoints = run_optimal_partitioning(
    X_complex_10_segments_n_200,
    cost_func,
    cost_init_func,
    penalty=pelt_penalty,
    min_segment_length=min_segment_length,
)
compute_segment_lengths(opt_part_changepoints, len(X_complex_10_segments_n_200))

np.testing.assert_array_almost_equal(new_opt_costs, opt_part_costs)
np.testing.assert_array_almost_equal(new_changepoints, opt_part_changepoints)

# %%
old_part_cost = pelt_partition_cost(
    X_complex_10_segments_n_200,
    old_changepoints,
    cost_func,
    cost_init_func,
    penalty=pelt_penalty,
)
new_part_cost = pelt_partition_cost(
    X_complex_10_segments_n_200,
    new_changepoints,
    cost_func,
    cost_init_func,
    penalty=pelt_penalty,
)
opt_part_cost = pelt_partition_cost(
    X_complex_10_segments_n_200,
    opt_part_changepoints,
    cost_func,
    cost_init_func,
    penalty=pelt_penalty,
)

# %%
print(
    f"Optimal partitioning cost:  {opt_part_cost:.4e} (F[-1]: {opt_part_costs[-1]:.4e})"
)
print(
    f"Old PELT partitioning cost: {old_part_cost:.4e} (F[-1]: {old_opt_costs[-1]:.4e})"
)
print(
    f"New PELT partitioning cost: {new_part_cost:.4e} (F[-1]: {new_opt_costs[-1]:.4e})"
)

# %%
px.line(x=range(len(new_opt_costs)), y=new_opt_costs)

# %%
px.line(x=range(len(opt_part_costs)), y=opt_part_costs)

# %%
px.line(x=range(len(old_opt_costs)), y=old_opt_costs)
