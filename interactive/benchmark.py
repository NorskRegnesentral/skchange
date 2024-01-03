from timeit import timeit

import numpy as np
import pandas as pd
import plotly.express as px

from skchange.anomaly_detectors.tests.test_anomaly_detectors import anomaly_detectors
from skchange.change_detectors.tests.test_change_detectors import change_detectors

detector_classes = anomaly_detectors + change_detectors
ns = [1000, 10000, 100000, 1000000]
n_runs = [100, 10, 1, 1]
timings = {}
for detector_class in detector_classes:
    detector_name = detector_class.__name__
    detector = detector_class()
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
# Capa                        0.018404  0.264217  2.493315  23.328677
# CircularBinarySegmentation  0.058446  0.618860  6.308148  64.935380
# MoscoreAnomaly              0.008403  0.112707  1.140971  11.919980
# Mvcapa                      0.015993  0.220780  2.258813  22.808392
# Moscore                     0.000631  0.000951  0.006055   0.065951
# Pelt                        0.004850  0.050476  0.831491  19.405542
# SeededBinarySegmentation    0.002788  0.021783  0.210576   2.070616
