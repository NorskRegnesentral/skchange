"""
Explores change detection using continuous linear trend cost function.

This module demonstrates how to use the MovingWindow detector with
ContinuousLinearTrendCost to identify changepoints in generated data.
"""

import matplotlib.pylab as plt
import numpy as np
import ruptures as rpt
import seaborn as sns
from ruptures.utils.utils import pairwise

from skchange.change_detectors import MovingWindow, SeededBinarySegmentation
from skchange.change_scores import BestFitLinearTrendScore, ContinuousLinearTrendScore
from skchange.costs import ContinuousLinearTrendCost, LinearTrendCost
from skchange.datasets import generate_alternating_data

continuous_linear_trend_cost = ContinuousLinearTrendCost.create_test_instance()

n_segments = 2
seg_len = 50
df = generate_alternating_data(
    n_segments=n_segments, mean=15, segment_length=seg_len, p=1, random_state=2
)

detector = MovingWindow(continuous_linear_trend_cost)
detector.fit(df)
continuous_linear_trend_changepoints = detector.predict(df)

change_scores = detector.transform_scores(df)

# Plot the changepoints and scores
# Set the style of seaborn
sns.set_theme(style="whitegrid")
# Create a figure with two subplots: one for data, one for scores
fig1, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)

# Plot the data on the upper axis
ax1.plot(df, label="Data", color="blue")
# Plot the changepoints on the upper axis
for cp in continuous_linear_trend_changepoints["ilocs"]:
    ax1.axvline(x=cp, color="red", linestyle="--", label="Changepoint")
ax1.set_ylabel("Value")
ax1.set_title("MovingWindow changepoint detection with ContinuousLinearTrendCost")
ax1.legend()

# Plot the change scores on the lower axis
ax2.plot(change_scores, label="Change Scores", color="green")
ax2.set_xlabel("Time")
ax2.set_ylabel("Score")
ax2.legend()

# Adjust layout
# plt.tight_layout()
# plt.show()
# %%
# Now do the same with LinearTrendCost for comparison
linear_trend_cost = LinearTrendCost.create_test_instance()

# Create detector with LinearTrendCost
cont_linear_trend_score = MovingWindow(linear_trend_cost)
cont_linear_trend_score.fit(df)
linear_trend_cost_changepoints = cont_linear_trend_score.predict(df)

cont_linear_trend_change_scores = cont_linear_trend_score.transform_scores(df)

# Create a new figure for LinearTrendCost
fig2, (ax3, ax4) = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)

# Plot the data on the upper axis
ax3.plot(df, label="Data", color="blue")
# Plot the changepoints on the upper axis
for cp in linear_trend_cost_changepoints["ilocs"]:
    ax3.axvline(x=cp, color="red", linestyle="--", label="Changepoint")
ax3.set_ylabel("Value")
ax3.set_title("MovingWindow changepoint detection with LinearTrendCost")
ax3.legend()

# Plot the change scores on the lower axis
ax4.plot(cont_linear_trend_change_scores, label="Change Scores", color="green")
ax4.set_xlabel("Time")
ax4.set_ylabel("Score")
ax4.legend()

# Adjust layout
# plt.tight_layout()
# plt.show()

# %%
# Draw the linear trends implied by the changepoints, for both costs.
# In the 'ContinuousLinearTrendCost' case, the linear trend is defined as the straight line
# connecting the first and last points of each interval.
# In the 'LinearTrendCost' case, the linear trend is defined as the best fit line
# through all points in the interval.
# This is done by fitting a linear regression model to the data in the interval.
from sklearn.linear_model import LinearRegression

# First, let's handle ContinuousLinearTrendCost trends
changepoints_with_endpoints = np.append(
    np.insert(continuous_linear_trend_changepoints["ilocs"], 0, 0), len(df)
)

for i in range(len(changepoints_with_endpoints) - 1):
    start = changepoints_with_endpoints[i]
    end = changepoints_with_endpoints[i + 1]

    # For ContinuousLinearTrendCost, draw line between first and last points
    x_vals = np.array([start, end - 1])
    y_vals = np.array([df.iloc[start].values[0], df.iloc[end - 1].values[0]])
    ax1.plot(
        x_vals,
        y_vals,
        color="orange",
        linestyle="-",
        linewidth=2,
        label="Trend" if i == 0 else "",
    )

# Now, handle LinearTrendCost trends
linear_changepoints_with_endpoints = np.append(
    np.insert(linear_trend_cost_changepoints["ilocs"], 0, 0), len(df)
)
for i in range(len(linear_changepoints_with_endpoints) - 1):
    start = linear_changepoints_with_endpoints[i]
    end = linear_changepoints_with_endpoints[i + 1]

    # For LinearTrendCost, fit a regression line through all points
    segment = np.arange(start, end)
    X = segment.reshape(-1, 1)
    y = df.iloc[start:end].values

    model = LinearRegression()
    model.fit(X, y)

    # Plot the regression line
    trend_line = model.predict(X)
    ax3.plot(
        segment,
        trend_line,
        color="orange",
        linestyle="-",
        linewidth=2,
        label="Trend" if i == 0 else "",
    )

# Update legends
handles1, labels1 = ax1.get_legend_handles_labels()
by_label1 = dict(zip(labels1, handles1))
ax1.legend(by_label1.values(), by_label1.keys())

handles3, labels3 = ax3.get_legend_handles_labels()
by_label3 = dict(zip(labels3, handles3))
ax3.legend(by_label3.values(), by_label3.keys())

fig1
fig2

# # Show the plots
# plt.tight_layout()
# plt.show()

# %%
# Now do the same with LinearTrendCost for comparison
cont_linear_trend_score = ContinuousLinearTrendScore()


# Create detector with LinearTrendCost
cont_linear_trend_score_detector = MovingWindow(
    cont_linear_trend_score
    # , bandwidth=30
)
cont_linear_trend_score_detector.fit(df)
cont_linear_trend_score_changepoints = cont_linear_trend_score_detector.predict(df)

cont_linear_trend_change_scores = cont_linear_trend_score_detector.transform_scores(df)

# Create a new figure for ContinuousLinearTrendScore
fig3, (ax5, ax6) = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)

# Plot the data on the upper axis
ax5.plot(df, label="Data", color="blue")
# Plot the changepoints on the upper axis
for cp in cont_linear_trend_score_changepoints["ilocs"]:
    ax5.axvline(x=cp, color="red", linestyle="--", label="Changepoint")
ax5.set_ylabel("Value")
ax5.set_title("MovingWindow changepoint detection with ContinuousLinearTrendScore")
ax5.legend()

# Plot the change scores on the lower axis
ax6.plot(cont_linear_trend_change_scores, label="Change Scores", color="green")
ax6.set_xlabel("Time")
ax6.set_ylabel("Score")
ax6.legend()

# Draw linear trends for ContinuousLinearTrendScore
score_changepoints_with_endpoints = np.append(
    np.insert(cont_linear_trend_score_changepoints["ilocs"], 0, 0), len(df)
)

for i in range(len(score_changepoints_with_endpoints) - 1):
    start = score_changepoints_with_endpoints[i]
    # Actually connect with the first point of the next segment:
    end = min(score_changepoints_with_endpoints[i + 1] + 1, len(df))

    # Draw line between first and last points
    x_vals = np.array([start, end - 1])
    y_vals = np.array([df.iloc[start].values[0], df.iloc[end - 1].values[0]])
    ax5.plot(
        x_vals,
        y_vals,
        color="orange",
        linestyle="-",
        linewidth=2,
        label="Trend" if i == 0 else "",
    )

# Update legend
handles5, labels5 = ax5.get_legend_handles_labels()
by_label5 = dict(zip(labels5, handles5))
ax5.legend(by_label5.values(), by_label5.keys())

fig3

# %%
# best_fit_linear_trend_score = BestFitLinearTrendScore()
best_fit_linear_trend_cost = 

# Create detector with LinearTrendCost
best_fit_linear_trend_score_mw_detector = MovingWindow(
    # best_fit_linear_trend_score
    best_fit_linear_trend_cost,
)
best_fit_linear_trend_score_mw_detector.fit(df)
best_fit_linear_trend_score_changepoints = (
    best_fit_linear_trend_score_mw_detector.predict(df)
)

best_fit_linear_trend_change_scores = (
    best_fit_linear_trend_score_mw_detector.transform_scores(df)
)

# Create a new figure for BestFitLinearTrendScore
fig4, (ax7, ax8) = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
# Plot the data on the upper axis
ax7.plot(df, label="Data", color="blue")
# Plot the changepoints on the upper axis
for cp in best_fit_linear_trend_score_changepoints["ilocs"]:
    ax7.axvline(x=cp, color="red", linestyle="--", label="Changepoint")
ax7.set_ylabel("Value")
ax7.set_title("MovingWindow changepoint detection with BestFitLinearTrendScore")
ax7.legend()
# Plot the change scores on the lower axis
ax8.plot(best_fit_linear_trend_change_scores, label="Change Scores", color="green")
ax8.set_xlabel("Time")
ax8.set_ylabel("Score")
ax8.legend()

# Fit piecewise linear trend using linear regression, with kink at each changepoint:
piecewise_linear_trend_regression_data = np.zeros(
    (len(df), 2 + len(best_fit_linear_trend_score_changepoints["ilocs"]))
)  # Intercept, slope1, slope2
piecewise_linear_trend_regression_data[:, 0] = 1.0  # Intercept
piecewise_linear_trend_regression_data[:, 1] = np.arange(len(df))  # Time steps
# Add columns for slopes at each changepoint
for i, cp in enumerate(best_fit_linear_trend_score_changepoints["ilocs"]):
    piecewise_linear_trend_regression_data[cp:, 2 + i] = np.arange(len(df) - cp)

# Fit the regression model
model = LinearRegression()
model.fit(
    piecewise_linear_trend_regression_data,
    df.values,
)
# Get the fitted values
best_fit_piecewise_linear_trend_fitted_values = model.predict(
    piecewise_linear_trend_regression_data
)
# Plot the fitted piecewise linear trend
ax7.plot(
    np.arange(len(df)),
    best_fit_piecewise_linear_trend_fitted_values,
    color="orange",
    linestyle="-",
    linewidth=2,
    label="Trend",
)

# Update legend
handles7, labels7 = ax7.get_legend_handles_labels()
by_label7 = dict(zip(labels7, handles7))
ax7.legend(by_label7.values(), by_label7.keys())
fig4

# Show the plots
# %%
# Use the SeededBinarySegmentation algorithm with the BestFitLinearTrendScore:
bf_linear_trend_seeded_bin_seg_detector = SeededBinarySegmentation(
    best_fit_linear_trend_score
)

# Fit the detector to the data
bf_linear_trend_seeded_bin_seg_detector.fit(df)
# Predict changepoints
bf_linear_trend_seeded_bin_seg_changepoints = (
    bf_linear_trend_seeded_bin_seg_detector.predict(df)
)

# Plot the data and changepoints
fig5, ax9 = plt.subplots(figsize=(12, 6))
ax9.plot(df, label="Data", color="blue")
# Plot the changepoints on the upper axis
for cp in best_fit_linear_trend_score_changepoints["ilocs"]:
    ax9.axvline(x=cp, color="red", linestyle="--", label="Changepoint")
ax9.set_ylabel("Value")
ax9.set_title(
    "SeededBinarySegmentation changepoint detection with BestFitLinearTrendScore"
)
ax9.legend()

# Fit piecewise linear trend using linear regression, with kink at each changepoint:
piecewise_linear_trend_regression_data = np.zeros(
    (len(df), 2 + len(bf_linear_trend_seeded_bin_seg_changepoints["ilocs"]))
)  # Intercept, slope1, slope2
piecewise_linear_trend_regression_data[:, 0] = 1.0  # Intercept
piecewise_linear_trend_regression_data[:, 1] = np.arange(len(df))  # Time steps
# Add columns for slopes at each changepoint
for i, cp in enumerate(bf_linear_trend_seeded_bin_seg_changepoints["ilocs"]):
    piecewise_linear_trend_regression_data[cp:, 2 + i] = np.arange(len(df) - cp)
# Fit the regression model
model = LinearRegression()
model.fit(
    piecewise_linear_trend_regression_data,
    df.values,
)
# Get the fitted values
best_fit_piecewise_linear_trend_fitted_values = model.predict(
    piecewise_linear_trend_regression_data
)
# Plot the fitted piecewise linear trend
ax9.plot(
    np.arange(len(df)),
    best_fit_piecewise_linear_trend_fitted_values,
    color="orange",
    linestyle="-",
    linewidth=2,
    label="Trend",
)
# Update legend
handles9, labels9 = ax9.get_legend_handles_labels()
by_label9 = dict(zip(labels9, handles9))
ax9.legend(by_label9.values(), by_label9.keys())

fig5
# Show the plot
# plt.tight_layout()
# plt.show()


# %% Test ruptures:
# creation of data
n_samples, n_dims = 500, 1  # number of samples, dimension
n_bkps, sigma = 3, 1.5  # number of change points, noise standard deviation
signal, bkps = rpt.pw_constant(n_samples, n_dims, n_bkps, noise_std=sigma, seed=421)
list(pairwise([0] + bkps))
signal = np.cumsum(signal, axis=1)

ruptures_clinear_cost = rpt.costs.CostCLinear().fit(signal)
print(ruptures_clinear_cost.error(50, 150))

print(ruptures_clinear_cost.sum_of_costs(bkps))
print(ruptures_clinear_cost.sum_of_costs([10, 100, 200, 250, n_samples]))

ruptures_clinear_cost = rpt.costs.CostCLinear()
algo = rpt.Dynp(custom_cost=ruptures_clinear_cost)
# is equivalent to
# algo = rpt.Dynp(model="clinear")
