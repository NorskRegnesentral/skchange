"""Interactive exploration of the MovingWindow change detector."""

import plotly.express as px

from skchange.change_detectors._moving_window import MovingWindow
from skchange.datasets import generate_alternating_data
from skchange.utils.benchmarking.profiler import Profiler

# Simple univariate example
df = generate_alternating_data(
    n_segments=2, mean=10, segment_length=100, p=1, random_state=2
)
detector = MovingWindow()
changepoints = detector.fit_predict(df)
labels = detector.transform(df)
scores = detector.transform_scores(df)
px.scatter(scores)


# Profiling
n = int(1e6)
df = generate_alternating_data(n_segments=1, mean=0, segment_length=n, p=1)
detector = MovingWindow(bandwidth=50)
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()


# Multiple bandwidths
df = generate_alternating_data(
    n_segments=2, mean=10, segment_length=100, p=1, random_state=2
)
detector = MovingWindow(bandwidth=[10, 20, 30, 40, 50, 60, 100])
changepoints = detector.fit_predict(df)
labels = detector.transform(df)
scores = detector.transform_scores(df)
px.scatter(scores)
