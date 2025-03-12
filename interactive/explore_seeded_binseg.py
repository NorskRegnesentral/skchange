"""Interactive exploration of Seeded Binary Segmentation."""

import plotly.express as px

from skchange.change_detectors import SeededBinarySegmentation
from skchange.datasets import generate_alternating_data
from skchange.utils.benchmarking.profiler import Profiler

df = generate_alternating_data(
    n_segments=2, mean=10, segment_length=20, p=1, random_state=7
)
detector = SeededBinarySegmentation(growth_factor=2)
detector.fit_predict(df)

px.line(df)
px.scatter(detector.scores, x="argmax_cpt", y="score", hover_data=["start", "end"])


# Profiling
n = int(1e6)
df = generate_alternating_data(n_segments=1, mean=0, segment_length=n, p=1)
detector = SeededBinarySegmentation(growth_factor=1.5, min_segment_length=10)
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()


# Test tuning
df_train = generate_alternating_data(
    n_segments=1, mean=0, segment_length=10000, p=1, random_state=9
)
df_test = generate_alternating_data(
    n_segments=10, mean=5, segment_length=1000, p=1, random_state=5
)
detector = SeededBinarySegmentation(threshold_scale=None, min_segment_length=10)
detector.fit(df_train)
changepoints = detector.predict(df_test)
