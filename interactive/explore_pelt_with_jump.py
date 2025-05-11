# %%

from skchange.change_detectors._crops import JumpCost
from skchange.change_detectors._pelt import (
    run_improved_pelt_array_based,
    run_pelt_with_jump,
)
from skchange.costs import GaussianCost
from skchange.datasets import generate_alternating_data

# %%
# Generate test data:
dataset = generate_alternating_data(
    n_segments=5,
    segment_length=100,
    p=1,
    mean=3.0,
    variance=4.0,
    random_state=42,
)

# cost = L2Cost()
cost = GaussianCost()
# # # cost = L1Cost()

test_penalty = 15.0
jump_step = 8

# %%
cost.fit(dataset.values)
pelt_with_jump_costs, pelt_with_jump_change_points = run_pelt_with_jump(
    cost,
    penalty=test_penalty,
    jump_step=jump_step,
)

# %%
# Run the same with a JumpCost:
jump_cost = JumpCost(GaussianCost(), jump=jump_step)
jump_cost.fit(dataset.values)
jump_cost_pelt_costs, jump_cost_change_points = run_improved_pelt_array_based(
    cost=jump_cost, penalty=test_penalty, min_segment_length=1
)
jump_cost_change_points *= jump_step

# %% TODO: Implement a Optimal Partitioning 'refinement' method that explores
# the neighborhood of the change points found by PELT with a jump step
# [change_point_u - (jump_step - 1), change_point_u + (jump_step - 1)].
# Then for CROPS, solve the problem with a jump step == min_segment_length,
# and then refine the change points.
