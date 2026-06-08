"""Interactive exploration of penalty_curve using the HVAC dataset."""

import numpy as np
import plotly.express as px

from skchange.new_api.datasets import load_hvac_system_data
from skchange.new_api.detectors import SeededBinarySegmentation
from skchange.new_api.tuning import penalty_curve

# ---------------------------------------------------------------------------
# Load data — use unit 1 only (single contiguous time series)
# ---------------------------------------------------------------------------
data = load_hvac_system_data()
unit_mask = data["unit_id"] == 1
X = data["data"][unit_mask]  # shape (n_samples, 1)
time = data["time"][unit_mask]

print(f"Unit 1: {X.shape[0]} samples, feature: {data['feature_names']}")
px.line(
    x=time, y=X[:, 0], labels={"x": "time", "y": "vibration"}, title="HVAC unit 1"
).show()

# ---------------------------------------------------------------------------
# Sweep the penalty parameter
# ---------------------------------------------------------------------------
detector = SeededBinarySegmentation()
penalty_range = np.logspace(-3, 2, 30)

counts = penalty_curve(
    detector,
    X,
    penalty_name="penalty",
    penalty_range=penalty_range,
)

# ---------------------------------------------------------------------------
# Figure 1: penalty curve (n_changepoints vs penalty)
# ---------------------------------------------------------------------------
fig_curve = px.line(
    x=penalty_range,
    y=counts,
    log_x=True,
    markers=True,
    labels={"x": "penalty (log scale)", "y": "n_changepoints"},
    title="Penalty curve — SeededBinarySegmentation on HVAC unit 1",
)
fig_curve.show()

# ---------------------------------------------------------------------------
# Figure 2: detected changepoints on the raw time series
# ---------------------------------------------------------------------------
target = 4 * 5 * 2  # 4 weeks, 5 working days, 2 changepoints per day
candidates = penalty_range[counts <= target]
selected = float(candidates.min()) if candidates.size else float(penalty_range[-1])
print(f"Selected penalty (≤ {target} changepoints): {selected:.2f}")

final = SeededBinarySegmentation(penalty=selected).fit(X)
cps = final.predict_changepoints(X)
print(f"Detected changepoints ({len(cps)}): {cps}")

fig_ts = px.line(
    x=time,
    y=X[:, 0],
    labels={"x": "time", "y": "vibration"},
    title=f"HVAC unit 1 — changepoints at penalty={selected:.2f} ({len(cps)} detected)",
)
for cp in cps:
    fig_ts.add_vline(
        x=str(time[cp].astype("datetime64[ms]")),
        line_dash="dash",
        line_color="red",
        opacity=0.6,
    )
fig_ts.show()
