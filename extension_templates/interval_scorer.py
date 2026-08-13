"""Extension template for new-API interval scorers.

Generic template for implementing a new interval scorer in
``skchange.interval_scorers``. Covers all four scorer types:

- **cost(start, end)**: Computes a cost over [start, end) intervals, with lower scores
  indicating better fit.
- **change_score(start, split, end)**: Computes a score/test statistic over
  [start, split, end) combinations, with higher scores indicating stronger evidence for
  a change at split within the interval [start, end].
- **saving(start, end)**: Computes a saving/test statistic by comparing a model fit to
  an interval [start, end) against a baseline model.
- **transient_score(outer_start, inner_start, inner_end, outer_end)**: Computes a
  score/test statistic for a transient/epidemic change
  interval [inner_start, inner_end) against the surrounding data defined by the
  union of [outer_start, inner_start) and [inner_end, outer_end).

How to use this template
------------------------
1. Make a copy of this file in a suitable location. For an internal extension,
   copy it to one of the following and rename to ``<your_scorer_name>.py``::

       skchange/interval_scorers/_costs/
       skchange/interval_scorers/_change_scores/
       skchange/interval_scorers/_savings/
       skchange/interval_scorers/_transient_scores/

2. Work through every "todo" comment below.

copyright: skchange developers, BSD-3-Clause License (see LICENSE file)
"""

from numbers import Real

import numpy as np
from sklearn.utils.validation import check_is_fitted

# todo: pick ONE base class from the four below and delete the others.
from skchange.interval_scorers._base import (
    BaseChangeScore,  # noqa: F401
    BaseCost,
    BaseSaving,  # noqa: F401
    BaseTransientScore,  # noqa: F401
)
from skchange.penalties import (
    bic_penalty,  # noqa: F401,  # todo: often needed for get_default_penalt, replace or delete as needed
)
from skchange.types import ArrayLike, Self
from skchange.utils._numeric import (
    col_cumsum,  # noqa: F401  # often handy, delete if not neeed
)
from skchange.utils._param_validation import (
    Interval,
    _fit_context,
)
from skchange.utils._tags import SkchangeTags
from skchange.utils.validation import (
    check_interval_specs,
    validate_data,
)

# External extension imports (when the file lives outside the skchange package),
# use these instead of the _base import above:
# from skchange.interval_scorers import (
#     BaseChangeScore,
#     BaseCost,
#     BaseSaving,
#     BaseTransientScore,
# )

# todo: replace BaseCost below with the base class matching your scorer type,
# and update the class docstring (description, math formula, parameter
# descriptions, references, and examples) to match.


class MyIntervalScorer(BaseCost):
    r"""Custom interval scorer.

    todo: write a docstring describing the scorer, including its scoring
    formula, when it is applicable, and any references.

    .. math::
        S([s, e)) = \\dots

    Parameters
    ----------
    param1 : float, default=1.0
        Descriptive explanation of ``param1``.
    param2 : str or None, default=None
        Descriptive explanation of ``param2``.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during ``fit``.
    n_samples_in_ : int
        Number of samples seen during ``fit``.

    Notes
    -----
    todo: optional notes about implementation tradeoffs, numerical
    considerations, references, etc.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.interval_scorers import MyIntervalScorer
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> scorer = MyIntervalScorer().fit(X)
    >>> cache = scorer.precompute(X)
    >>> scorer.evaluate(cache, np.array([[0, 50], [50, 100]]))  # doctest: +SKIP
    """

    # ------------------------------------------------------------------
    # Optional: __init__ and _parameter_constraints
    # ------------------------------------------------------------------
    # Only needed if the scorer has hyper-parameters. Delete both the
    # _parameter_constraints block and __init__ if there are no hyper-parameters.
    # If kept, follow sklearn conventions in __init__:
    #  - one argument per public hyper-parameter
    #  - store each argument under the same name on self without
    #    transformation; do not write derived state here
    #  - do NOT call super().__init__(); sklearn's BaseEstimator handles it
    _parameter_constraints: dict = {
        "param1": [Interval(Real, 0, None, closed="left")],
        "param2": [str, None],
    }

    def __init__(
        self,
        param1: float = 1.0,
        param2: str | None = None,
    ):
        self.param1 = param1
        self.param2 = param2

    # ------------------------------------------------------------------
    # Optional: __sklearn_tags__
    # ------------------------------------------------------------------
    # Override only if the scorer deviate from the default tags.
    # See skchange/utils/_tags.py for the full list of available tags
    # (SkchangeInputTags, IntervalScorerTags) and their semantics and
    # defaults. Delete this method entirely if no overrides are needed.
    def __sklearn_tags__(self) -> SkchangeTags:
        """Return scorer tags."""
        tags = super().__sklearn_tags__()
        # tags.interval_scorer_tags.aggregated = True
        # tags.interval_scorer_tags.penalised = True
        # tags.interval_scorer_tags.non_negative_scores = False
        # tags.input_tags.integer_only = True
        # tags.input_tags.conditional = True
        # tags.input_tags.multivariate = False
        return tags

    # ------------------------------------------------------------------
    # Optional: fit
    # ------------------------------------------------------------------
    # Override only when the scorer has hyper-parameters to validate and/or
    # learns quantities from the full training array (e.g. a baseline mean,
    # thresholds). For stateless scorers (e.g. L2Cost, CUSUM), delete this
    # method entirely — the base class default validates X and records
    # n_samples_in_ / n_features_in_.
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the scorer to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored. Included for API consistency.

        Returns
        -------
        self : MyIntervalScorer
            Fitted scorer.
        """
        # IMPORTANT: validate_data records n_features_in_ and n_samples_in_
        # on self (when reset=True).
        # IMPORTANT: avoid side effects to X / y.
        X = validate_data(self, X, ensure_2d=True, reset=True)

        # todo: write additional parameter validation and fitting logic here.
        # todo: write derived state to attributes ending in an underscore.
        # Example baseline for a saving:
        # self.baseline_mean_ = np.median(X, axis=0)

        return self

    # ------------------------------------------------------------------
    # Strongly recommended: precompute
    # ------------------------------------------------------------------
    # Override to precompute statistics that make evaluate O(1) per interval
    # (e.g. cumulative sums, cumulative sums of squares, rank tables).
    # Delete this method if the base class default {"X": X} is sufficient.
    def precompute(self, X: ArrayLike) -> dict:
        """Precompute statistics that speed up ``evaluate``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute over. Typically the same array passed to
            ``fit``, but may differ (e.g. when computing scores on a
            validation set).

        Returns
        -------
        cache : dict
            Dictionary of precomputed arrays consumed by ``evaluate``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)

        # todo: replace with the precomputation appropriate for your scorer.
        # For example for an L2-style scorer:
        cache = {
            "sums": col_cumsum(X, init_zero=True),
            "sums2": col_cumsum(X**2, init_zero=True),
        }
        return cache

    # ------------------------------------------------------------------
    # Mandatory: evaluate
    # ------------------------------------------------------------------
    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate the scorer on a batch of interval specifications.

        Parameters
        ----------
        cache : dict
            Output of :meth:`precompute`.
        interval_specs : array-like of shape (n_interval_specs, ncols)
            Interval boundaries. The expected number of columns depends on
            the scorer type and equals ``self.interval_specs_ncols``:

            - cost / saving (``ncols = 2``): ``[start, end)``
            - change score (``ncols = 3``): ``[start, split, end)``
            - transient score (``ncols = 4``):
              ``[outer_start, inner_start, inner_end, outer_end)``

        Returns
        -------
        scores : ndarray
            If the scorer is feature-wise, shape is
            ``(n_interval_specs, n_features)`` with one column per input
            feature. If the scorer is aggregated
            (``interval_scorer_tags.aggregated = True``), shape is
            ``(n_interval_specs, 1)``.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )

        # todo: unpack columns appropriately for the scorer type. Example for
        # a cost / saving:
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        # For a change score:
        # starts, splits, ends = (
        #     interval_specs[:, 0], interval_specs[:, 1], interval_specs[:, 2]
        # )
        # For a transient score:
        # outer_s, inner_s, inner_e, outer_e = (
        #     interval_specs[:, 0],
        #     interval_specs[:, 1],
        #     interval_specs[:, 2],
        #     interval_specs[:, 3],
        # )

        # todo: implement the vectorised scoring formula. The output MUST be
        # 2D — one row per interval and either n_features columns
        # (feature-wise scorers) or 1 column (aggregated scorers).
        sums = cache["sums"]
        n = (ends - starts).reshape(-1, 1)
        return (sums[ends] - sums[starts]) ** 2 / n  # placeholder formula

    # ------------------------------------------------------------------
    # Strongly recommended: get_default_penalty
    # ------------------------------------------------------------------
    # Needed for automatic penalty selection (e.g. PELT with penalty=None,
    # or BIC selection inside CROPS). Most costs return a scalar BIC penalty;
    # some savings return a per-feature array via mvcapa_penalty. Delete this
    # method entirely if the scorer should not support a default penalty.
    def get_default_penalty(self) -> float | np.ndarray:
        """Return the default penalty for the fitted scorer.

        Returns
        -------
        float or np.ndarray
            Default penalty value or array.
        """
        check_is_fitted(self)
        return bic_penalty(self.n_samples_in_, self.n_features_in_)

    # ------------------------------------------------------------------
    # Optional: min_size
    # ------------------------------------------------------------------
    # Override only if the scorer requires more than 1 sample per subinterval in
    # evaluate(). Delete otherwise.
    @property
    def min_size(self) -> int:
        """Minimum number of samples required per subinterval in evaluate.

        The subinterval sizes are calculated from the interval_specs columns as
        ``np.diff(interval_specs, axis=1)``. E.g., for a cost/saving, the subinterval
        size is ``end - start``; for a change score, the subinterval sizes are
        ``split - start`` and ``end - split``, etc.
        """
        # Examples:
        # - mean + variance: return 2
        # - covariance matrix (data-dependent):
        #     check_is_fitted(self)
        #     return self.n_features_in_ + 1
        return 1


# todo: for internal extensions, after implementing the scorer also:
#  1. Export the class from
#     skchange/interval_scorers/__init__.py — add the import in
#     alphabetic order within its scorer-type block and the name to __all__.
#  2. Add an unfitted instance of the class to the appropriate list
#     (_COSTS / _CHANGE_SCORES / _SAVINGS / _TRANSIENT_SCORES) in
#     skchange/interval_scorers/tests/_registry.py. This is how your
#     scorer is exercised by the common tests in
#     skchange/interval_scorers/tests/test_all.py.
#  3. If your scorer is a BaseCost that is NOT subadditive under the
#     concatenated-surrounding baseline (required by CostTransientScore), add
#     its class name to CostTransientScore._INCOMPATIBLE_COST_NAMES in
#     skchange/interval_scorers/_from_cost.py.
#  4. Optional: Add a dedicated test file under
#     skchange/interval_scorers/tests/ for any scorer-specific
#     behaviour not covered by test_all.py.
