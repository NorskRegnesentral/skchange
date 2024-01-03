import numpy as np
import pandas as pd
import plotly.express as px
from streamchange.utils import Profiler

from skchange.anomaly_detectors.moscore_anomaly import run_moscore_anomaly
from skchange.datasets.generate import generate_anomalous_data
from skchange.scores.mean_score import init_mean_score, mean_anomaly_score

n = 500
df = generate_anomalous_data(
    n, anomalies=[(100, 119), (250, 299)], means=[10.0, 5.0], random_state=1
)
px.scatter(df)

anomaly_lengths = np.arange(10, 100)
anomalies, scores = run_moscore_anomaly(
    df.values, mean_anomaly_score, init_mean_score, anomaly_lengths, 50, 50, 20.0
)
scores = pd.DataFrame(scores, index=df.index, columns=anomaly_lengths)
px.line(scores)

# Profiling
n = int(1e5)
df = generate_anomalous_data(n, anomalies=[(100, 119), (250, 299)], means=[10.0, 5.0])

profiler = Profiler()
profiler.start()
anomalies, scores = run_moscore_anomaly(
    df.values, mean_anomaly_score, init_mean_score, anomaly_lengths, 50, 50, 20.0
)
profiler.stop()


# Check mean anomaly score
n = 500
df = generate_anomalous_data(n, anomalies=[(100, 119), (250, 299)], means=[10.0, 5.0])
df = df.iloc[:40]

params = init_mean_score(df.values)
starts = np.arange(10, 20)
ends = starts + 10 - 1
background_starts = starts - 10
background_ends = ends + 10
scores = mean_anomaly_score(params, background_starts, background_ends, starts, ends)
