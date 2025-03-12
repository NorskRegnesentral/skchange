"""Interacive exploration of the Circular Binary Segmentation anomaly detector."""

import plotly.express as px

from skchange.anomaly_detectors._circular_binseg import (
    CircularBinarySegmentation,
    make_anomaly_intervals,
)
from skchange.datasets import generate_alternating_data
from skchange.utils.benchmarking.profiler import Profiler

df = generate_alternating_data(
    n_segments=3, mean=10, segment_length=20, p=1, random_state=7
)
detector = CircularBinarySegmentation(growth_factor=1.5, min_segment_length=10)
anomalies = detector.fit_predict(df)
px.line(df)
px.scatter(detector.scores, x="argmax_anomaly_start", y="score")

# Test anomaly intervals
anomaly_intervals = make_anomaly_intervals(0, 5, 2)

# Profiling
n = int(1e3)
df = generate_alternating_data(n_segments=1, mean=0, segment_length=n, p=1)
detector = CircularBinarySegmentation(
    threshold_scale=4.0, max_interval_length=100, growth_factor=2
)
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()
