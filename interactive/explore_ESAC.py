"""Interactive exploration of Seeded Binary Segmentation with ESAC penalty."""

import plotly.express as px

from skchange.change_detectors import SeededBinarySegmentation
from skchange.change_scores import ESACScore
from skchange.datasets import generate_alternating_data
from skchange.utils.benchmarking.profiler import Profiler

df = generate_alternating_data(
    n_segments=2, mean=10, segment_length=20, p=1, random_state=7
)
detector = SeededBinarySegmentation(growth_factor=2, change_score=ESACScore())
print(detector.fit_predict(df))
print("")
print("a_s: ", detector.fitted_score.a_s)
print("nu_s: ", detector.fitted_score.nu_s)
print("t_s: ", detector.fitted_score.t_s)
print("threshold: ", detector.fitted_score.threshold)
print("armaxes sparsity: ", detector.fitted_score.sargmaxes)


px.line(df)
px.scatter(detector.scores, x="argmax", y="max", hover_data=["start", "end"])


## check sparsity
df = generate_alternating_data(
    n_segments=10,
    mean=10,
    segment_length=20,
    p=1000,
    random_state=7,
    affected_proportion=0.001,
)
detector = SeededBinarySegmentation(growth_factor=2, change_score=ESACScore())
print(detector.fit_predict(df))
print("")
print("a_s: ", detector.fitted_score.a_s)
print("nu_s: ", detector.fitted_score.nu_s)
print("t_s: ", detector.fitted_score.t_s)
print("threshold: ", detector.fitted_score.threshold)
print("armaxes sparsity: ", detector.fitted_score.sargmaxes)
print("n : ", detector.fitted_score.n)
print("p : ", detector.fitted_score.p)

px.line(df)
px.scatter(detector.scores, x="argmax", y="max", hover_data=["start", "end"])


# Profiling
n = int(1e4)
df = generate_alternating_data(n_segments=1, mean=0, segment_length=n, p=100)
detector = SeededBinarySegmentation(growth_factor=2, change_score=ESACScore())
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()
