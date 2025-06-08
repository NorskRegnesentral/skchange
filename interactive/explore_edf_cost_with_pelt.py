"""Script demonstrating the use of the Empirical Distribution Cost."""

# %%
import matplotlib.pyplot as plt

from skchange.change_detectors import PELT
from skchange.costs import EmpiricalDistributionCost
from skchange.datasets import generate_alternating_data

alternating_data = generate_alternating_data(
    n_segments=2,
    segment_length=100,
    p=1,
    random_state=42,
    mean=2.5,
    variance=2.5,
)

# %% Plot the generated data:
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(alternating_data, label="Generated Data")
ax.set_title("Generated Alternating Data")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
plt.show()

# %%
cost = EmpiricalDistributionCost(use_cache=True)
detector = PELT(cost=cost, min_segment_length=15, penalty=15.0)
changepoints = detector.fit_predict(alternating_data)["ilocs"]
print("Detected changepoints:", changepoints)


# %% Plot the detected changepoints:
fig, axes = plt.subplots(figsize=(10, 5), nrows=3, ncols=1, sharex=True, sharey=True)
axes[0].ecdf(alternating_data.to_numpy().reshape(-1), label="ECDF of Data")

first_segment = alternating_data.iloc[: changepoints[0]]
axes[1].ecdf(first_segment.to_numpy().reshape(-1), label="ECDF of First Segment")
second_segment = alternating_data.iloc[changepoints[0] :]
axes[2].ecdf(second_segment.to_numpy().reshape(-1), label="ECDF of Second Segment")

plt.show()

# %%
