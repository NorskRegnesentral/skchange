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
    """

    multivariate: bool = True
    integer_only: bool = False
    conditional: bool = False


@dataclass(slots=True)
class ChangeDetectorTags:
    """Tags specific to change detection estimators.

    Attributes
    ----------
    variable_identification : bool, default=False
        Whether the detector can identify which variables are affected
        at each changepoint.
    """

    variable_identification: bool = False


@dataclass(slots=True)
class IntervalScorerTags:
    """Tags specific to interval scorer estimators.

    Attributes
    ----------
    score_type : str | None, default=None
        Type of score: "cost", "change_score", "saving", "transient_score", None
    aggregated : bool, default=False
        Whether the scorer always returns a single value per cut,
        irrespective of input data shape.
    penalised : bool, default=False
        Whether the score is inherently penalised. If True, score > 0
        indicates change/anomaly. If False, external penalisation needed.
    fixed_model : bool, default=False
        Whether all model parameters in the scorer are fixed.
    """

    score_type: str | None = None
    aggregated: bool = False
    penalised: bool = False


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
