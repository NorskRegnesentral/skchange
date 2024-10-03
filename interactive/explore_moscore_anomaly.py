"""Interactive exploration of the MoscoreAnomaly detector."""

import numpy as np
import plotly.express as px

from skchange.anomaly_detectors.moscore_anomaly import MoscoreAnomaly
from skchange.datasets.generate import generate_anomalous_data
from skchange.scores.mean_score import init_mean_score, mean_anomaly_score
from skchange.utils.benchmarking.profiler import Profiler

n = 500
df = generate_anomalous_data(
    n, anomalies=[(100, 119), (250, 299)], means=[10.0, 5.0], random_state=1
)
px.scatter(df)

detector = MoscoreAnomaly(
    score="mean_var",
    min_anomaly_length=10,
    max_anomaly_length=100,
    threshold_scale=3.0,
    left_bandwidth=50,
)
anomalies = detector.fit_predict(df)
print(anomalies)


# Profiling
n = int(1e5)
df = generate_anomalous_data(n, anomalies=[(100, 119), (250, 299)], means=[10.0, 5.0])
detector = MoscoreAnomaly(
    score="mean",
    min_anomaly_length=10,
    max_anomaly_length=100,
    left_bandwidth=50,
)

profiler = Profiler()
profiler.start()
anomalies = detector.fit_predict(df)
profiler.stop()


# Check mean anomaly score
n = 500
df = generate_anomalous_data(n, anomalies=[(100, 119), (250, 299)], means=[10.0, 5.0])
df = df.iloc[:40]

params = init_mean_score(df.values)
starts = np.arange(10, 20)
ends = starts + 10 - 1
background_starts = starts - 10
background_ends = ends + 10
scores = mean_anomaly_score(params, background_starts, background_ends, starts, ends)
