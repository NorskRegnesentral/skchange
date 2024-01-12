import plotly.express as px

from skchange.anomaly_detectors.capa import Capa
from skchange.anomaly_detectors.mvcapa import Mvcapa
from skchange.datasets.generate import generate_teeth_data
from skchange.utils.benchmarking.profiler import Profiler

# Unviariate
df = generate_teeth_data(n_segments=5, mean=10, segment_length=10, p=1, random_state=2)
capa = Capa(fmt="sparse", max_segment_length=20)
anomalies = capa.fit_predict(df)

capa = Capa(labels="score", fmt="dense", max_segment_length=20)
scores = capa.fit_predict(df)

capa = Capa(labels="indicator", fmt="dense", max_segment_length=20)
anomalies = capa.fit_predict(df)
px.scatter(x=df.index, y=df.values[:, 0], color=anomalies)

# Multivariate
# TODO: Add plotting functionality to assess the affected subset.
df = generate_teeth_data(5, 10, p=10, mean=10, affected_proportion=0.2, random_state=2)
capa = Mvcapa(collective_penalty="sparse", fmt="sparse")
anomalies = capa.fit_predict(df)

capa = Mvcapa(labels="score", fmt="dense", max_segment_length=20)
scores = capa.fit_predict(df)

capa = Mvcapa(collective_penalty_scale=5, labels="indicator", fmt="dense")
anomalies = capa.fit_predict(df)
df.plot(kind="line", backend="plotly")
anomalies.plot(kind="line", backend="plotly")


# Profiling
n = int(1e5)
df = generate_teeth_data(n_segments=1, mean=0, segment_length=n, p=1)
detector = Capa(
    max_segment_length=100, collective_penalty_scale=5, point_penalty_scale=5
)
detector = Mvcapa(
    max_segment_length=1000,
    collective_penalty="sparse",
    collective_penalty_scale=5,
    point_penalty_scale=5,
)
profiler = Profiler().start()
detector.fit_predict(df)
profiler.stop()
