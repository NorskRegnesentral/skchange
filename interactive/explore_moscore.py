import numpy as np
import plotly.express as px
from streamchange.utils import Profiler

from skchange.change_detectors.moscore import Moscore, where
from skchange.datasets.generate import teeth
from skchange.scores.mean_score import init_mean_score, mean_score

# Compare skchange output to streamchange
df = teeth(n_segments=2, mean=10, segment_length=100, p=1, random_state=2)
detector = Moscore()
changepoints = detector.fit_predict(df)
px.scatter(detector.scores)


# Profiling
n = int(1e6)
df = teeth(n_segments=1, mean=0, segment_length=n, p=1)
detector = Moscore(bandwidth=50)
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()


# Various unit tests
df = teeth(n_segments=1, mean=10, segment_length=10, p=1)
precomputed_params = init_mean_score(df.values)
mean_score(precomputed_params, start=0, end=9, split=4)
where(np.array([True, True, True, False, False]))


# Variance score
df = teeth(n_segments=2, variance=16, segment_length=100, p=1, random_state=1)
detector = Moscore(score="var")
changepoints = detector.fit_predict(df)
px.scatter(df)
px.scatter(detector.scores)
