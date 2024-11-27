"""Benchmarking the computational efficiency of the detectors."""

from timeit import timeit

import numpy as np
import pandas as pd
import plotly.express as px

from skchange.anomaly_detectors import ANOMALY_DETECTORS
from skchange.change_detectors import CHANGE_DETECTORS

# TODO: Add all the different scores and costs.
# TODO: Make sure hyperparameters are set such that comparisons are fair.
detector_classes = ANOMALY_DETECTORS + CHANGE_DETECTORS
ns = [1000, 10000, 100000, 1000000]
n_runs = [100, 10, 1, 1]
timings = {}
for detector_class in detector_classes:
    detector_name = detector_class.__name__
    detector = detector_class.create_test_instance()
    setup_data = pd.DataFrame(np.random.normal(0, 1, size=1000))
    detector.fit_predict(setup_data)  # Compile numba
    timings[detector_name] = []
    for n, n_run in zip(ns, n_runs):
        df = pd.DataFrame(np.random.normal(0, 1, size=n))
        timing = timeit(
            "detector.fit_predict(df)",
            number=n_run,
            globals=globals(),
        )
        mean_timing = timing / n_run
        timings[detector_name].append(mean_timing)

timings = pd.DataFrame(timings, index=pd.Index(ns, name="n"))

timings_long = timings.melt(
    ignore_index=False, value_name="execution_time", var_name="detector"
).reset_index()
px.line(
    timings_long,
    x="n",
    y="execution_time",
    color="detector",
    log_x=True,
    log_y=True,
    title="Running times (s) of detectors",
)

# Current results
# n                            1000      10000     100000     1000000
# CAPA                        0.007719  0.069377  0.663148   6.679747
# CircularBinarySegmentation  0.029423  0.296935  3.052292  34.491029
# MovingWindowAnomaly              0.010520  0.123790  1.182320  13.913373
# MVCAPA                      0.010347  0.072321  0.717383   6.754314
# StatThresholdAnomaliser     0.001964  0.002588  0.011953   0.118650
# MovingWindow                     0.000711  0.001072  0.006007   0.068569
# PELT                        0.004251  0.067712  0.925646  23.362470
# SeededBinarySegmentation    0.002616  0.019799  0.187402   1.899995
