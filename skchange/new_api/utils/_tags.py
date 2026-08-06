from dataclasses import dataclass, field

from sklearn.utils import InputTags, Tags, TargetTags


@dataclass(slots=True)
class SkchangeInputTags(InputTags):
    """Extended input tags for skchange estimators.

    Extends sklearn's InputTags with additional skchange-specific input constraints.

    Attributes
    ----------
    multivariate : bool, default=True
        Whether the estimator can handle multivariate data (n_features > 1).
    integer_only : bool, default=False
        Whether the estimator requires integer-valued input data (e.g., for count data).
    conditional : bool, default=False
        Whether the estimator uses some input columns as covariates. If True,
        at least two input columns are required (one response, one+ covariates).
    timestamps : bool, default=False
        Whether the estimator expects one column of the input to contain timestamps.
        **Experimental**: support for timestamp columns is not yet stable and may
        change in future releases.
    """

    multivariate: bool = True
    integer_only: bool = False
    conditional: bool = False
    timestamps: bool = False


@dataclass(slots=True)
class ChangeDetectorTags:
    """Tags specific to change detection estimators.

    Attributes
    ----------
    linear_trend_segment : bool, default=False
        Whether the detector is designed for data where each segment follows a
        linear trend. When ``True``, test fixtures will generate piecewise linear
        data with a kink at the changepoint rather than a mean shift.
    calibration_strategy : str, default="detection_count"
        How FWER calibration computes the per-null-sample critical penalty
        scale for this detector.

        - ``"detection_count"`` (default): bisect ``penalty_scale`` until the
          number of detections hits zero. Exact for any detector with a single
          ``penalty_scale`` knob. No structural assumption required.
        - ``"max_score"``: closed-form ``c_b = max(S) / base``. Exact for
          scan-and-threshold detectors that threshold each interval score
          independently (e.g. SeededBinarySegmentation, MovingWindow,
          CircularBinarySegmentation).
        - ``"path_search"``: exact secant search on the convex hull of
          cost-vs-number-of-changepoints. Used by PELT because the
          single-split ``max_score`` underestimates its true critical
          penalty.
    """

    linear_trend_segment: bool = False
    calibration_strategy: str = "detection_count"


@dataclass(slots=True)
class IntervalScorerTags:
    """Tags specific to interval scorer estimators.

    Attributes
    ----------
    score_type : str | None, default=None
        Type of score: "cost", "change_score", "saving", "transient_score", None
    aggregated : bool, default=False
        Whether the scorer always returns a single value per ``interval_spec``,
        irrespective of input data shape.
    penalised : bool, default=False
        Whether the score is inherently penalised. If True, score > 0
        indicates change/anomaly. If False, external penalisation needed.
    non_negative_scores : bool, default=True
        Whether the scorer is guaranteed to return non-negative values on
        homogeneous data. Most costs and savings satisfy this. Set to False
        for costs that are test statistics which can be <= 0 by construction
        (e.g. ``RankCost``).
    linear_trend_segment : bool, default=False
        Whether the scorer is designed for data where each segment follows a
        linear trend. When ``True``, test fixtures will generate piecewise linear
        data with a kink at the changepoint rather than a mean shift.
    """

    score_type: str | None = None
    aggregated: bool = False
    penalised: bool = False
    non_negative_scores: bool = True
    linear_trend_segment: bool = False


@dataclass
class SkchangeTags(Tags):
    """Extended tags for skchange estimators.

    Extends sklearn's base Tags with change detection specific tag groups.

    Attributes
    ----------
    input_tags : SkchangeInputTags
        Extended input data tags with skchange-specific constraints.
    change_detector_tags : ChangeDetectorTags
        Change detection specific tags.
    interval_scorer_tags : IntervalScorerTags
        Interval scorer specific tags.
    """

    # Re-declare required parent fields with defaults
    # Must be declared at the top of the class to avoid dataclass ordering issues.
    estimator_type: str | None = None
    target_tags: TargetTags = field(default_factory=lambda: TargetTags(required=False))
    input_tags: SkchangeInputTags = field(default_factory=SkchangeInputTags)

    # New fields.
    # The presence of one of these tag classes indicate the type of estimator.
    change_detector_tags: ChangeDetectorTags | None = None
    interval_scorer_tags: IntervalScorerTags | None = None
