"""Interactive exploration of the StatThresholdAnomaliser."""

import numpy as np

from skchange.anomaly_detectors import StatThresholdAnomaliser
from skchange.change_detectors import PELT, MovingWindow
from skchange.datasets import generate_anomalous_data

n = 500
df = generate_anomalous_data(
    n, anomalies=[(100, 119), (250, 299)], means=[10.0, 5.0], random_state=1
)

change_detector = MovingWindow(bandwidth=20)
change_detector = PELT(min_segment_length=5)
detector = StatThresholdAnomaliser(
    change_detector, stat=np.mean, stat_lower=-1.0, stat_upper=1.0
)
anomalies = detector.fit_predict(df)
print(anomalies)
