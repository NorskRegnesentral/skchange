import numpy as np
import pandas as pd
from streamchange.offline.costs import OfflineL2Cost
from streamchange.offline.pelt import OfflinePelt
from streamchange.penalties import ConstantPenalty

from skchange.change_detectors.pelt import BIC_penalty, Pelt
from skchange.datasets.generate import teeth

# Generate data
df = teeth(n_segments=2, mean_size=10, segment_length=100000, p=5, random_state=2)

# streamchange method
detector = OfflinePelt(
    OfflineL2Cost(),
    penalty=ConstantPenalty(BIC_penalty(df.shape[0], df.shape[1])),
    minsl=2,
)
detector.fit(df)
print(detector.segments_)

# skchange method
detector = Pelt()
detector.fit_predict(df)


# Profiling
from streamchange.utils import Profiler

df = teeth(n_segments=2, mean_size=10, segment_length=100000, p=1)
detector = Pelt()
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()

detector = Pelt(OfflineL2Cost(), minsl=2, maxsl=1000000)
segments = detector.fit_predict(df)
print(pd.DataFrame(segments))


# Various unit tests
from skchange.costs.l2_cost import init_l2_cost, l2_cost

df = teeth(n_segments=3, mean_size=10, segment_length=5, p=2)
precomputed_params = init_l2_cost(df.values)
l2_cost(precomputed_params, starts=np.array([1, 2, 3]), ends=np.array([3, 4, 5]))
detector = Pelt()
detector.fit_predict(df)
