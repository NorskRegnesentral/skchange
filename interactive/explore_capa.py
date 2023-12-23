import plotly.express as px
from streamchange.capa import Capa as StreamchangeCapa
from streamchange.capa import ConstMeanL2
from streamchange.utils import Profiler

from skchange.anomaly_detectors.capa import Capa
from skchange.datasets.generate import teeth

# Unviariate, compare to streamchange
# streamchange
df = teeth(n_segments=4, mean=10, segment_length=10, p=1, random_state=2)
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
capa = Capa(collective_penalty_scale=5, fmt="sparse", max_segment_length=20)
anomalies = capa.fit_predict(df)

capa = Capa(labels="score", fmt="dense", max_segment_length=20)
scores = capa.fit_predict(df)

capa = Capa(labels="indicator", fmt="dense", max_segment_length=20)
anomalies = capa.fit_predict(df)
df.plot(kind="scatter", backend="plotly")
px.scatter(x=df.index, y=df.values[:, 0], color=anomalies)


# Profiling
# TODO: Add a dedicated univariate version. Currently Capa is x10 slower than strchange
# TODO: Add pruning?
n = int(1e4)
df = teeth(n_segments=1, mean=0, segment_length=n, p=1)
detector = Capa(
    max_segment_length=1000, collective_penalty_scale=5, point_penalty_scale=5
)
streamchange_capa = StreamchangeCapa(
    ConstMeanL2(), maxsl=1000, predict_point_anomalies=True
)
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()
