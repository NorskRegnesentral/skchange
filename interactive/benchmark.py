from timeit import timeit

import numpy as np
import pandas as pd
import plotly.express as px

from skchange.anomaly_detectors.tests.test_anomaly_detectors import anomaly_detectors
from skchange.change_detectors.tests.test_change_detectors import change_detectors


def print_with_time(text):
    now_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_time}] {text}")


detector_classes = anomaly_detectors + change_detectors
detector_classes = [change_detectors[0]]
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
        print_with_time(f"n={n}, detector={detector_name}")
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
# Capa	CircularBinarySegmentation	MoscoreAnomaly	Mvcapa	Moscore	Pelt	SeededBinarySegmentation
# n
# 1000	0.013922	0.060348	0.009355	0.018739	0.001471	0.005443	0.005522
# 10000	0.269205	0.666049	0.128670	0.287116	0.004255	0.071429	0.048885
# 100000	2.723129	6.877902	1.234485	2.961893	0.045651	1.535496	0.492033
# 1000000	27.328482	69.811003	13.031418	29.076900	0.320187	32.254520	4.902390
