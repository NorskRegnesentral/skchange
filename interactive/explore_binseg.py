import plotly.express as px
from streamchange.utils import Profiler

from skchange.change_detectors.binary_segmentation import SeededBinarySegmentation
from skchange.datasets.generate import teeth

df = teeth(n_segments=2, mean=10, segment_length=20, p=1, random_state=7)
detector = SeededBinarySegmentation(score="mean", growth_factor=2)
detector.fit_predict(df)

df.plot(kind="line", backend="plotly")

px.scatter(detector.scores, x="maximizer", y="score", hover_data=["start", "end"])


# Profiling
n = int(1e6)
df = teeth(n_segments=1, mean=0, segment_length=n, p=1)
detector = SeededBinarySegmentation("mean", growth_factor=1.5)
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()
