import plotly.express as px

from skchange.anomaly_detectors.circular_binseg import (
    CircularBinarySegmentation,
    make_anomaly_intervals,
)
from skchange.datasets.generate import generate_teeth_data
from skchange.utils.benchmarking.profiler import Profiler

df = generate_teeth_data(n_segments=3, mean=10, segment_length=20, p=1, random_state=7)
detector = CircularBinarySegmentation(
    score="mean", growth_factor=1.5, min_segment_length=10
)
anomalies = detector.fit_predict(df)

df.plot(kind="line", backend="plotly")

px.scatter(detector.scores, x="argmax_anomaly_start", y="score")

# Test anomaly intervals
anomaly_intervals = make_anomaly_intervals(0, 5, 2)

# Profiling
n = int(1e5)
df = generate_teeth_data(n_segments=1, mean=0, segment_length=n, p=1)
detector = CircularBinarySegmentation("mean", growth_factor=2)
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()
