"""Interactive exploration of the PELT change detector."""

from skchange.change_detectors import PELT
from skchange.datasets import generate_alternating_data
from skchange.utils.benchmarking.profiler import Profiler

# Simple univariate example
df = generate_alternating_data(
    n_segments=2, mean=10, segment_length=100, p=1, random_state=2
)
detector = PELT(min_segment_length=1)
detector.fit(df)
detector.predict(df)
detector.transform(df)
detector.transform_scores(df)


# Profiling
n = int(1e6)
df = generate_alternating_data(n_segments=1, mean=0, segment_length=n, p=1)
detector = PELT()
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()
