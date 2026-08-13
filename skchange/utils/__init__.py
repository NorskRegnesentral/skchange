"""General purpose utility functions."""

from skchange.utils._tags import (
    ChangeDetectorTags,
    IntervalScorerTags,
    SkchangeInputTags,
    SkchangeTags,
)
from skchange.utils.plotting import (
    plot_detections,
    plot_segmentation,
)
from skchange.utils.segmentation import (
    changepoints_to_labels,
    labels_to_changepoints,
)
from skchange.utils.validation import (
    check_interval_scorer,
    check_interval_specs,
    check_penalty,
    validate_data,
)

__all__ = [
    "ChangeDetectorTags",
    "IntervalScorerTags",
    "SkchangeInputTags",
    "SkchangeTags",
    "check_interval_scorer",
    "check_interval_specs",
    "check_penalty",
    "changepoints_to_labels",
    "labels_to_changepoints",
    "plot_detections",
    "plot_segmentation",
    "validate_data",
]
