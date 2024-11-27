"""Interactive exploration of the CAPA and Mvcapa anomaly detectors."""

import pandas as pd
import plotly.express as px

from skchange.anomaly_detectors.capa import CAPA
from skchange.anomaly_detectors.mvcapa import Mvcapa
from skchange.datasets.generate import generate_alternating_data
from skchange.utils.benchmarking.profiler import Profiler

# Unviariate
df = generate_alternating_data(
    n_segments=5, segment_length=10, mean=10, random_state=2
)[0]
detector = CAPA(max_segment_length=20)

anomalies = detector.fit_predict(df)
print(anomalies)

anomaly_labels = detector.fit_transform(df)
px.scatter(x=df.index, y=df, color=anomaly_labels.astype(str))

scores = detector.score_transform(df)
px.scatter(scores)

# Multivariate
df = generate_alternating_data(
    5, 10, p=10, mean=10, affected_proportion=0.2, random_state=2
)
detector = Mvcapa(collective_penalty="sparse")

anomalies = detector.fit_predict(df)
print(anomalies)

anomaly_labels = detector.fit_transform(df)
anomaly_labels = (anomaly_labels > 0).astype(int)
anomaly_labels[anomaly_labels == 0] = 0.1
plot_df = pd.concat(
    [
        df.melt(ignore_index=False).reset_index(),
        anomaly_labels.melt(value_name="anomaly_label")["anomaly_label"],
    ],
    axis=1,
)
plot_df["variable"] = plot_df["variable"].astype(str)
px.scatter(plot_df, x="index", y="value", color="variable", size="anomaly_label")

fig = px.line(df)
fig.add_scatter(anomaly_labels)
px.line(anomaly_labels)

scores = detector.score_transform(df)
px.scatter(scores)


# Profiling
n = int(1e5)
df = generate_alternating_data(n_segments=1, mean=0, segment_length=n, p=1)
detector = CAPA(
    max_segment_length=100, collective_penalty_scale=5, point_penalty_scale=5
)
detector = Mvcapa(
    max_segment_length=1000,
    collective_penalty="sparse",
    collective_penalty_scale=5,
    point_penalty_scale=5,
)
profiler = Profiler().start()
detector.fit_predict(df)
profiler.stop()
