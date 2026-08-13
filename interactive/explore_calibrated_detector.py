"""Verify and explore the CalibratedDetector README example.

Demonstrates penalty calibration for SeededBinarySegmentation on beta-distributed
data with two changepoints, using separate change-free data for the null sampler.
"""

import scipy.stats as st

from skchange.datasets import generate_piecewise_data
from skchange.detectors import SeededBinarySegmentation
from skchange.tuning import CalibratedDetector
from skchange.utils.plotting import plot_detections

# Change-free beta(2, 5) data used to calibrate the detection threshold.
X_calib = generate_piecewise_data(st.beta(2, 5), lengths=300, seed=0)

# Test data: two changepoints where the beta shape flips.
X = generate_piecewise_data(
    [st.beta(2, 5), st.beta(5, 2), st.beta(1, 10)],
    lengths=100,
    seed=1,
)

detector = CalibratedDetector(
    SeededBinarySegmentation(),
    level=0.05,
    n_simulations=999,
    random_state=0,
)
detector.fit(X_calib)
changepoints = detector.predict(X)

print("Calibrated penalty_scale:", detector.penalty_scale_)
print("Detected changepoints:   ", changepoints)

fig = plot_detections(X, changepoints=changepoints)
fig.show()
