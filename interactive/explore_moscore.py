import numpy as np
import plotly.express as px
from numba import njit
from streamchange.utils import Profiler

from skchange.change_detectors.moscore import Moscore, where
from skchange.datasets.generate import add_linspace_outliers, generate_teeth_data
from skchange.scores.mean_score import init_mean_score, mean_score

# Compare skchange output to streamchange
df = generate_teeth_data(n_segments=2, mean=10, segment_length=100, p=1, random_state=2)
detector = Moscore()
changepoints = detector.fit_predict(df)
px.scatter(detector.scores)


# Profiling
n = int(1e6)
df = generate_teeth_data(n_segments=1, mean=0, segment_length=n, p=1)
detector = Moscore(bandwidth=50)
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()


# Various unit tests
df = generate_teeth_data(n_segments=1, mean=10, segment_length=10, p=1)
precomputed_params = init_mean_score(df.values)
mean_score(precomputed_params, start=0, end=9, split=4)
where(np.array([True, True, True, False, False]))


# Variance score
df = generate_teeth_data(
    n_segments=2, variance=16, segment_length=100, p=1, random_state=1
)
detector = Moscore(score="meanvar")
changepoints = detector.fit_predict(df)
px.scatter(df)
px.scatter(detector.scores)


# Custom score
@njit
def col_median(X: np.ndarray) -> np.ndarray:
    m = X.shape[1]
    medians = np.zeros(m)
    for j in range(m):
        medians[j] = np.median(X[:, j])
    return medians


@njit
def init_spike_score(X: np.ndarray) -> np.ndarray:
    return X


def spike_score_factory(margin: int = 0):
    @njit
    def spike_score(
        precomputed_params: np.ndarray, start: int, end: int, split: int
    ) -> float:
        X = precomputed_params
        interval_X = np.concatenate(
            (X[start : split - margin], X[split + margin + 1 : end + 1])
        )
        baseline_median = col_median(interval_X)
        return np.sum(np.abs(X[split] - baseline_median))

    return spike_score


df = generate_teeth_data(n_segments=1, mean=0, segment_length=100, p=1)
df = add_linspace_outliers(df, n_outliers=4, outlier_size=10)
score = (spike_score_factory(margin=0), init_spike_score)
detector = Moscore(score, bandwidth=5)
anomalies = detector.fit_predict(df)
px.scatter(detector.scores)
px.scatter(df)
