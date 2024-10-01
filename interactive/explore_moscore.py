"""Interactive exploration of the Moscore change detector."""

import numpy as np
import plotly.express as px
from numba import njit

from skchange.change_detectors.moscore import Moscore
from skchange.datasets.generate import add_linspace_outliers, generate_teeth_data
from skchange.utils.benchmarking.profiler import Profiler

# Simple univariate example
df = generate_teeth_data(n_segments=2, mean=10, segment_length=100, p=1, random_state=2)
detector = Moscore()
changepoints = detector.fit_predict(df)
labels = detector.transform(df)
scores = detector.score_transform(df)
px.scatter(scores)


# Profiling
n = int(1e6)
df = generate_teeth_data(n_segments=1, mean=0, segment_length=n, p=1)
detector = Moscore(bandwidth=50)
profiler = Profiler()
profiler.start()
detector.fit_predict(df)
profiler.stop()


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
    """Compute the median of each column of X."""
    m = X.shape[1]
    medians = np.zeros(m)
    for j in range(m):
        medians[j] = np.median(X[:, j])
    return medians


@njit
def init_spike_score(X: np.ndarray) -> np.ndarray:
    """Initialize the spike score."""
    return X


@njit
def spike_score(
    precomputed_params: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    split: np.ndarray,
) -> float:
    """Calculate the score for a spike at the split point."""
    X = precomputed_params
    baseline_median = np.zeros((len(start), X.shape[1]))
    for i, (s, e) in enumerate(zip(start, end)):
        baseline_median[i] = col_median(X[s : e + 1])
    return np.sum(np.abs(X[split] - baseline_median), axis=1)


df = generate_teeth_data(n_segments=1, mean=0, segment_length=100, p=1)
df = add_linspace_outliers(df, n_outliers=4, outlier_size=10)
score = (spike_score, init_spike_score)
detector = Moscore(score, bandwidth=5)
anomalies = detector.fit_predict(df)
px.scatter(detector.scores)
px.scatter(df)
