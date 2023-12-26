import plotly.express as px
from streamchange.capa import Capa as StreamchangeCapa
from streamchange.capa import ConstMeanL2
from streamchange.utils import Profiler

from skchange.anomaly_detectors.capa import Capa
from skchange.anomaly_detectors.mvcapa import Mvcapa
from skchange.datasets.generate import teeth

# Unviariate, compare to streamchange
# streamchange
df = teeth(n_segments=5, mean=10, segment_length=10, p=1, random_state=2)
streamchange_capa = StreamchangeCapa(
    ConstMeanL2(), minsl=2, maxsl=20, predict_point_anomalies=True
)
anomalies = streamchange_capa.fit_predict(df)

# skchange
capa = Capa(fmt="sparse", max_segment_length=20)
anomalies = capa.fit_predict(df)

capa = Capa(labels="score", fmt="dense", max_segment_length=20)
scores = capa.fit_predict(df)

capa = Capa(labels="indicator", fmt="dense", max_segment_length=20)
anomalies = capa.fit_predict(df)
px.scatter(x=df.index, y=df.values[:, 0], color=anomalies)

# Multivariate
# TODO: Add plotting functionality to assess the affected subset.
df = teeth(5, 10, p=10, mean=10, affected_proportion=0.2, random_state=2)
capa = Mvcapa(
    collective_penalty_scale=3,
    collective_penalty="sparse",
    fmt="sparse",
)
anomalies = capa.fit_predict(df)

capa = Mvcapa(labels="score", fmt="dense", max_segment_length=20)
scores = capa.fit_predict(df)

capa = Mvcapa(collective_penalty_scale=5, labels="indicator", fmt="dense")
anomalies = capa.fit_predict(df)
df.plot(kind="line", backend="plotly")
anomalies.plot(kind="line", backend="plotly")


# Profiling
# TODO: Find optimal subset in Mvcapa after detecting temporal position of anomaly.
#       Waste of resources to find and store all optimisers along the way.
#       Also more accurate to always rerun with 'sparse' penalty.
# TODO: Add pruning?
n = int(1e5)
df = teeth(n_segments=1, mean=0, segment_length=n, p=1)
detector = Capa(
    max_segment_length=1000, collective_penalty_scale=5, point_penalty_scale=5
)
detector = Mvcapa(
    max_segment_length=1000,
    collective_penalty="sparse",
    collective_penalty_scale=5,
    point_penalty_scale=5,
)
detector = StreamchangeCapa(ConstMeanL2(), maxsl=1000, predict_point_anomalies=True)
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()
