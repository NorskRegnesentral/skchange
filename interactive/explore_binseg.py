import plotly.express as px
from streamchange.utils import Profiler

from skchange.change_detectors.seeded_binseg import SeededBinarySegmentation
from skchange.datasets.generate import teeth

df = teeth(n_segments=2, mean=10, segment_length=20, p=1, random_state=7)
detector = SeededBinarySegmentation(score="mean", growth_factor=2)
detector.fit_predict(df)

df.plot(kind="line", backend="plotly")

px.scatter(detector.scores, x="maximizer", y="score", hover_data=["start", "end"])


# Profiling
n = int(1e6)
df = teeth(n_segments=1, mean=0, segment_length=n, p=1)
detector = SeededBinarySegmentation("mean", growth_factor=1.5, min_segment_length=10)
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()


# Test tuning
df_train = teeth(n_segments=1, mean=0, segment_length=10000, p=1, random_state=9)
df_test = teeth(n_segments=10, mean=5, segment_length=1000, p=1, random_state=5)
detector = SeededBinarySegmentation(
    score="mean", threshold_scale=None, min_segment_length=10
)
detector.fit(df_train)
changepoints = detector.predict(df_test)
