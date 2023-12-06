import numpy as np
from streamchange.offline.costs import OfflineL2Cost
from streamchange.offline.pelt import OfflinePelt
from streamchange.penalties import ConstantPenalty
from streamchange.utils import Profiler

from skchange.change_detectors.pelt import BIC_penalty, Pelt
from skchange.costs.l2_cost import init_l2_cost, l2_cost
from skchange.datasets.generate import teeth

# Compare skchange output to streamchange
df = teeth(n_segments=2, mean=10, segment_length=100, p=1, random_state=2)
streamchange_detector = OfflinePelt(
    OfflineL2Cost(),
    penalty=ConstantPenalty(BIC_penalty(df.shape[0], df.shape[1])),
    minsl=2,
)
streamchange_detector.fit(df)
streamchange_detector.segments_
detector = Pelt()
detector.fit_predict(df)


# Profiling
n = int(1e6)
df = teeth(n_segments=1, mean=0, segment_length=n, p=1)
detector = Pelt()
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()

detector = Pelt(OfflineL2Cost(), minsl=2, maxsl=1000000)
segments = detector.fit_predict(df)


# Various unit tests
df = teeth(n_segments=3, mean=10, segment_length=5, p=2)
precomputed_params = init_l2_cost(df.values)
l2_cost(precomputed_params, starts=np.array([1, 2, 3]), ends=np.array([3, 4, 5]))
detector = Pelt()
detector.fit_predict(df)
