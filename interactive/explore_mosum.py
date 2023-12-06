import plotly.express as px
from streamchange.utils import Profiler

from skchange.change_detectors.mosum import Mosum
from skchange.datasets.generate import teeth
from skchange.test_stats.mean_test_stat import init_mean_test_stat, mean_test_stat

# Compare skchange output to streamchange
df = teeth(n_segments=2, mean=10, segment_length=100, p=1, random_state=2)
detector = Mosum()
scores = detector.fit_predict(df)
px.scatter(scores)


# Profiling
n = int(1e6)
df = teeth(n_segments=1, mean=0, segment_length=n, p=1)
detector = Mosum(bandwidth=50)
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()


# Various unit tests
df = teeth(n_segments=1, mean=10, segment_length=10, p=1)
precomputed_params = init_mean_test_stat(df.values)
mean_test_stat(precomputed_params, start=0, end=9, split=4)
