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
# Capa                        0.008452  0.069429  0.664815   6.581158
# CircularBinarySegmentation  0.028880  0.285940  2.899422  27.964504
# MoscoreAnomaly              0.008229  0.112742  1.153293  11.794399
# Mvcapa                      0.010236  0.096589  0.936205   9.430577
# Moscore                     0.000657  0.000973  0.006668   0.065385
# Pelt                        0.004611  0.050660  0.931237  24.531638
# SeededBinarySegmentation    0.002960  0.023154  0.223794   2.228483
