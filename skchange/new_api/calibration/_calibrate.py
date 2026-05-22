"""Core calibration function for false alarm control."""

import numpy as np
from sklearn.base import clone

from skchange.new_api.calibration._null_models import (
    BaseDataSampler,
    BaseParametricSampler,
)
from skchange.new_api.interval_scorers._base import BaseIntervalScorer


def _make_conservative_interval_specs(
    n_samples: int, scorer: BaseIntervalScorer
) -> np.ndarray:
    """Generate a conservative set of interval specs covering many splits.

    Uses a step of ``max(1, n_samples // 100)`` so the number of specs stays
    manageable even for large n.

    Parameters
    ----------
    n_samples : int
        Length of the series.
    scorer : BaseIntervalScorer
        Fitted scorer; used to determine ``interval_specs_ncols`` and ``min_size``.

    Returns
    -------
    interval_specs : np.ndarray of shape (n_specs, ncols)
    """
    ncols = scorer.interval_specs_ncols
    min_size = scorer.min_size
    max_half = n_samples // 2
    step = max(1, n_samples // 100)

    if ncols == 2:
        # [start, end) — savings / costs
        rows = []
        for length in range(min_size, max_half + 1, step):
            for start in range(0, n_samples - length + 1, step):
                rows.append([start, start + length])
        return (
            np.array(rows, dtype=np.int64) if rows else np.empty((0, 2), dtype=np.int64)
        )

    elif ncols == 3:
        # [start, split, end) — change scores
        rows = []
        for start in range(0, n_samples - 2 * min_size + 1, step):
            for end in range(
                start + 2 * min_size, min(n_samples + 1, start + max_half + 1), step
            ):
                for split in range(start + min_size, end - min_size + 1, step):
                    rows.append([start, split, end])
        if not rows:
            return np.empty((0, 3), dtype=np.int64)
        return np.array(rows, dtype=np.int64)

    elif ncols == 4:
        # [outer_start, inner_start, inner_end, outer_end) — transient scores.
        # Use the full series as the outer window; vary inner start/end positions.
        # A single large outer window is conservative: the background estimate is
        # stable, and the score is maximised over all possible inner intervals.
        rows = []
        for inner_start in range(min_size, n_samples - 2 * min_size + 1, step):
            for inner_end in range(
                inner_start + min_size, n_samples - min_size + 1, step
            ):
                rows.append([0, inner_start, inner_end, n_samples])
        if not rows:
            return np.empty((0, 4), dtype=np.int64)
        return np.array(rows, dtype=np.int64)

    else:
        raise ValueError(
            f"Unsupported interval_specs_ncols={ncols} for conservative default."
        )


def _max_score_ratio(
    scorer: BaseIntervalScorer,
    X: np.ndarray,
    interval_specs: np.ndarray,
    base_penalty: np.ndarray,
) -> float:
    """Compute max_interval [raw_score / base_penalty] on data X.

    For a vector base_penalty the penalised score formula is
    max_k [cumsum_sorted_k - base_penalty_k], and we want the multiplier c
    such that the max penalised score ≈ 0 when penalty = c * base_penalty.

    For scalar base_penalty: c* = max_interval raw_score / base_penalty_scalar.
    For vector base_penalty: we binary-search for c* such that
    max_interval max_k[cumsum_sorted_k - c * base_penalty_k] = 0.
    We implement this via direct computation of the raw scores and
    solve analytically: c* = max_interval max_k[cumsum_sorted_k / base_penalty_k].
    """
    cache = scorer.precompute(X)
    raw_scores = scorer.evaluate(cache, interval_specs)  # (n_specs, n_features)
    raw_scores = np.asarray(raw_scores, dtype=np.float64)

    penalty_arr = np.asarray(base_penalty, dtype=np.float64).reshape(-1)

    if penalty_arr.size == 1:
        # Constant penalty: penalised score = sum(features) - penalty.
        # c* = max_interval sum(features) / penalty.
        interval_sums = raw_scores.sum(axis=1)
        return float(np.max(interval_sums) / penalty_arr[0])

    # Vector: for each interval, compute max_k [cumsum_sorted_k / penalty_k]
    # Each row of raw_scores is the per-feature score for that interval.
    max_ratio = -np.inf
    p = penalty_arr.size
    for i in range(len(raw_scores)):
        row = raw_scores[i]
        # Sort descending; compute cumsum / penalty_k
        sorted_row = np.sort(row)[::-1]
        cumsum = np.cumsum(sorted_row)
        ratios = cumsum / penalty_arr[:p]
        r = float(np.max(ratios))
        if r > max_ratio:
            max_ratio = r
    return max_ratio


def calibrate_penalty_scale(
    scorer: BaseIntervalScorer,
    X: np.ndarray,
    sampler: "BaseDataSampler | BaseParametricSampler",
    interval_specs: np.ndarray | None = None,
    detector=None,
    level: float = 0.05,
    n_simulations: int = 999,
    random_state=None,
    X_calib: np.ndarray | None = None,
    X_train: np.ndarray | None = None,
) -> float:
    """Calibrate a penalty *scale* to achieve a target false alarm level.

    Runs ``n_simulations`` Monte Carlo simulations under the null, computes the
    maximum score/penalty ratio on each, and returns the
    ``(1 - level)``-quantile.  This is the scale factor ``c_star`` such that
    setting ``penalty = c_star * base_penalty`` controls the false alarm rate
    at the desired ``level``.

    Use :func:`calibrate_penalty` when you want the absolute calibrated
    penalty directly.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        An unpenalised interval scorer. Fitted on ``X`` internally.
    X : np.ndarray of shape (n_samples, n_features)
        Data to be analysed for changes. Determines the penalty scale
        (via ``n_samples``) and is used to fit the scorer for the base penalty.
    sampler : BaseDataSampler or BaseParametricSampler
        Null sampler.  Data-based samplers (:class:`BaseDataSampler`) are
        fitted on ``X_calib`` when provided, otherwise on ``X``, and their
        :meth:`sample` is called as ``sample(n_samples, rng)``.  Parametric
        samplers (:class:`BaseParametricSampler`) are never fitted; their
        :meth:`sample` is called as ``sample(n_samples, n_features, rng)``.
    interval_specs : np.ndarray or None, default=None
        Explicit interval specifications to evaluate. Highest priority.
    detector : BaseChangeDetector or None, default=None
        If provided and has ``get_interval_specs(n_samples)``, those intervals
        are used (second priority). Ignored when ``interval_specs`` is given.
    level : float, default=0.05
        Target false alarm probability.
    n_simulations : int, default=999
        Number of Monte Carlo simulations.
    random_state : int, Generator, or None, default=None
        Seed for reproducibility.
    X_calib : np.ndarray of shape (n_calib, n_features) or None, default=None
        Optional separate null (change-free) dataset used to fit data-based
        samplers.  Ignored for parametric samplers.  When ``None``, data-based
        samplers are fitted on ``X_train`` when provided, otherwise on ``X``.
    X_train : np.ndarray of shape (n_train, n_features) or None, default=None
        Optional larger training dataset.  When provided, the scorer is fitted
        on ``X_train`` (for better parameter estimates) instead of ``X``.  Also
        used as the fallback null-fit dataset for data-based samplers when
        ``X_calib`` is not provided.  ``len(X)`` always determines the null
        sample length and interval specs regardless of ``X_train``.

    Returns
    -------
    c_star : float
        Calibrated penalty scale factor.  Multiply by
        ``scorer.get_default_penalty()`` to obtain the absolute penalty.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"`X` must be 2-D, got shape {X.shape}.")
    n_samples, n_features = X.shape

    if X_calib is not None:
        X_calib = np.asarray(X_calib, dtype=np.float64)
        if X_calib.ndim != 2:
            raise ValueError(f"`X_calib` must be 2-D, got shape {X_calib.shape}.")
        if X_calib.shape[1] != X.shape[1]:
            raise ValueError(
                f"`X_calib` has {X_calib.shape[1]} features but `X` has "
                f"{X.shape[1]}. They must match."
            )

    if X_train is not None:
        X_train = np.asarray(X_train, dtype=np.float64)
        if X_train.ndim != 2:
            raise ValueError(f"`X_train` must be 2-D, got shape {X_train.shape}.")
        if X_train.shape[1] != X.shape[1]:
            raise ValueError(
                f"`X_train` has {X_train.shape[1]} features but `X` has "
                f"{X.shape[1]}. They must match."
            )

    # Fit scorer on X_train when provided (better parameter estimates), else X.
    # n_samples from X always determines null sample length and interval specs.
    scorer_fitted = clone(scorer).fit(X_train if X_train is not None else X)
    base_penalty = scorer_fitted.get_default_penalty()
    base_penalty_arr = np.asarray(base_penalty, dtype=np.float64).reshape(-1)

    # Fit data-based sampler on null data; parametric samplers need no fitting.
    # Priority: X_calib > X_train > X.
    if isinstance(sampler, BaseDataSampler):
        null_fit_data = X_calib if X_calib is not None else (X_train if X_train is not None else X)
        sampler_ready = clone(sampler).fit(null_fit_data)

        def _draw_null(rng: np.random.Generator) -> np.ndarray:
            return sampler_ready.sample(n_samples, rng)

    elif isinstance(sampler, BaseParametricSampler):

        def _draw_null(rng: np.random.Generator) -> np.ndarray:
            return sampler.sample(n_samples, n_features, rng)

    else:
        raise TypeError(
            f"'sampler' must be a BaseDataSampler or BaseParametricSampler instance, "
            f"got {type(sampler).__name__!r}."
        )

    # Resolve interval specifications (priority: explicit > detector > conservative).
    if interval_specs is not None:
        specs = np.asarray(interval_specs, dtype=np.int64)
    elif detector is not None and hasattr(detector, "get_interval_specs"):
        specs = detector.get_interval_specs(n_samples)
    else:
        specs = _make_conservative_interval_specs(n_samples, scorer_fitted)

    if len(specs) == 0:
        raise ValueError(
            "No valid interval specs could be generated for calibration. "
            "Ensure n_samples is large enough relative to scorer.min_size."
        )

    # RNG setup.
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    # Monte Carlo: collect max ratio per simulation.
    null_ratios = np.empty(n_simulations, dtype=np.float64)
    for b in range(n_simulations):
        X_null = _draw_null(rng)
        # Use scorer_fitted (trained on X) to evaluate each null sample.
        # Re-fitting on X_null would give a different baseline (e.g. a different
        # mean in L2Saving), which mismatches the actual detection setup where the
        # scorer is always trained on X.  Keeping scorer_fitted ensures the null
        # distribution in calibration matches the null distribution in detection.
        null_ratios[b] = _max_score_ratio(
            scorer_fitted, X_null, specs, base_penalty_arr
        )

    return float(np.quantile(null_ratios, 1.0 - level))


def calibrate_penalty(
    scorer: BaseIntervalScorer,
    X: np.ndarray,
    sampler: "BaseDataSampler | BaseParametricSampler",
    interval_specs: np.ndarray | None = None,
    detector=None,
    level: float = 0.05,
    n_simulations: int = 999,
    random_state=None,
    X_calib: np.ndarray | None = None,
    X_train: np.ndarray | None = None,
) -> "float | np.ndarray":
    """Calibrate a penalty to achieve a target false alarm level.

    Convenience wrapper around :func:`calibrate_penalty_scale` that returns
    the absolute calibrated penalty (``c_star * base_penalty``) rather than
    the scale factor ``c_star`` alone.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        An unpenalised interval scorer. Fitted on ``X`` internally.
    X : np.ndarray of shape (n_samples, n_features)
        Data to be analysed for changes.
    sampler : BaseDataSampler or BaseParametricSampler
        Null sampler.  See :func:`calibrate_penalty_scale` for details.
    interval_specs : np.ndarray or None, default=None
        Explicit interval specifications to evaluate. Highest priority.
    detector : BaseChangeDetector or None, default=None
        If provided and has ``get_interval_specs(n_samples)``, those intervals
        are used (second priority). Ignored when ``interval_specs`` is given.
    level : float, default=0.05
        Target false alarm probability.
    n_simulations : int, default=999
        Number of Monte Carlo simulations.
    random_state : int, Generator, or None, default=None
        Seed for reproducibility.
    X_calib : np.ndarray of shape (n_calib, n_features) or None, default=None
        Optional separate null dataset for data-based samplers.
    X_train : np.ndarray of shape (n_train, n_features) or None, default=None
        Optional larger training dataset.  See :func:`calibrate_penalty_scale`.

    Returns
    -------
    penalty : float or np.ndarray
        Calibrated penalty. Shape matches ``scorer.get_default_penalty()``.
    """
    c_star = calibrate_penalty_scale(
        scorer=scorer,
        X=X,
        sampler=sampler,
        interval_specs=interval_specs,
        detector=detector,
        level=level,
        n_simulations=n_simulations,
        random_state=random_state,
        X_calib=X_calib,
        X_train=X_train,
    )
    scorer_fitted = clone(scorer).fit(X_train if X_train is not None else X)
    base_penalty = scorer_fitted.get_default_penalty()
    base_penalty_arr = np.asarray(base_penalty, dtype=np.float64).reshape(-1)

    if np.ndim(base_penalty) == 0 or (
        isinstance(base_penalty, np.ndarray) and base_penalty.ndim == 0
    ):
        return float(c_star * float(base_penalty))

    result = c_star * base_penalty_arr
    if isinstance(base_penalty, np.ndarray):
        return result
    return float(result[0])
