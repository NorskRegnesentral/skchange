"""Interactive exploration of the CAPA anomaly detector."""

import plotly.express as px

from skchange.anomaly_detectors import CAPA
from skchange.datasets import generate_alternating_data

# Unviariate
df = generate_alternating_data(
    n_segments=5, segment_length=10, mean=10, random_state=2
)[0]
detector = CAPA(max_segment_length=20)

anomalies = detector.fit_predict(df)
print(anomalies)

anomaly_labels = detector.fit_transform(df)
px.scatter(x=df.index, y=df, color=anomaly_labels.astype(str))

scores = detector.transform_scores(df)
px.scatter(scores)
