"""Interactively compare the outputs of the different detector types."""

import numpy as np
import plotly.express as px

from skchange.anomaly_detectors import CAPA
from skchange.anomaly_scores import L2Saving
from skchange.change_detectors import MovingWindow
from skchange.datasets import generate_anomalous_data
from skchange.penalties import make_linear_chi2_penalty

# Generate data
n = 300
means = [np.array([8.0, 0.0, 0.0]), np.array([2.0, 3.0, 5.0])]
df = generate_anomalous_data(
    n, anomalies=[(100, 120), (250, 300)], means=means, random_state=3
)
df.columns += 1
plot_df = (
    df.stack()
    .reset_index()
    .rename({"level_0": "time", "level_1": "variable", 0: "value"}, axis=1)
)
fig = px.line(plot_df, x="time", y="value", facet_row="variable")
fig.show()

# Change detector
change_detector = MovingWindow(threshold_scale=1.0)
changepoints = change_detector.fit_predict(df)
changepoint_labels = change_detector.transform(df)
print(changepoints)
print(changepoint_labels)

# Segment anomaly detector
anomaly_detector = CAPA()
anomalies = anomaly_detector.fit_predict(df)
anomaly_labels = anomaly_detector.transform(df)
print(anomalies)
print(anomaly_labels)

# Subset segment anomaly detector
score = L2Saving()
p = df.shape[1]
penalty = make_linear_chi2_penalty(score.get_model_size(p), n, p)
subset_anomaly_detector = CAPA(score, penalty, find_affected_components=True)
subset_anomalies = subset_anomaly_detector.fit_predict(df)
subset_anomaly_labels = subset_anomaly_detector.transform(df)
print(subset_anomalies)
print(subset_anomaly_labels)
