import numpy as np

from skchange.change_detectors.pelt import Pelt
from skchange.costs.mean_cost import init_mean_cost, mean_cost
from skchange.datasets.generate import generate_teeth_data
from skchange.utils.benchmarking.profiler import Profiler

# Simple univariate example
df = generate_teeth_data(n_segments=2, mean=10, segment_length=100, p=1, random_state=2)
detector = Pelt(min_segment_length=1)
detector.fit_predict(df)


# Profiling
n = int(1e6)
df = generate_teeth_data(n_segments=1, mean=0, segment_length=n, p=1)
detector = Pelt()
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()


# Various unit tests
df = generate_teeth_data(n_segments=3, mean=10, segment_length=5, p=2)
precomputed_params = init_mean_cost(df.values)
mean_cost(precomputed_params, starts=np.array([1, 2, 3]), ends=np.array([3, 4, 5]))
detector = Pelt()
detector.fit_predict(df)
