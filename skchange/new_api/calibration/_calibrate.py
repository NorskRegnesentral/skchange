"""Core calibration function for false alarm control."""

import numpy as np
from sklearn.base import clone

from skchange.new_api.calibration._null_models import BaseNullModel
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


def calibrate_penalty(
    scorer: BaseIntervalScorer,
    X: np.ndarray,
    null_model: BaseNullModel,
    interval_specs: np.ndarray | None = None,
    detector=None,
    level: float = 0.05,
    n_simulations: int = 999,
    random_state=None,
    X_calib: np.ndarray | None = None,
) -> float | np.ndarray:
    """Calibrate a penalty to achieve a target false alarm level.

    Runs ``n_simulations`` Monte Carlo simulations under the null, computes the
    maximum score/penalty ratio on each, and returns the
    ``(1 - level)``-quantile multiplied by the base penalty.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        An unpenalised interval scorer. Fitted on X internally.
    X : np.ndarray of shape (n_samples, n_features)
        Data to be analysed for changes. Determines the penalty scale
        (via ``n_samples``) and is used to fit the scorer for the base penalty.
    null_model : BaseNullModel
        Null model. Fitted on ``X_calib`` when provided, otherwise on ``X``.
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
        Optional separate null (change-free) dataset used to fit the null model.
        Can be any length. When ``None``, the null model is fitted on ``X``.
        Providing this avoids contaminating the null model with changepoints that
        may be present in ``X``.

    Returns
    -------
    penalty : float or np.ndarray
        Calibrated penalty. Shape matches ``scorer.get_default_penalty()``.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"`X` must be 2-D, got shape {X.shape}.")
    n_samples = X.shape[0]

    if X_calib is not None:
        X_calib = np.asarray(X_calib, dtype=np.float64)
        if X_calib.ndim != 2:
            raise ValueError(f"`X_calib` must be 2-D, got shape {X_calib.shape}.")
        if X_calib.shape[1] != X.shape[1]:
            raise ValueError(
                f"`X_calib` has {X_calib.shape[1]} features but `X` has "
                f"{X.shape[1]}. They must match."
            )

    # Fit scorer on X to obtain the base penalty at the correct n_samples scale.
    # (get_default_penalty depends on n_samples_in_, which must equal len(X).)
    scorer_fitted = clone(scorer).fit(X)
    base_penalty = scorer_fitted.get_default_penalty()
    base_penalty_arr = np.asarray(base_penalty, dtype=np.float64).reshape(-1)

    # Fit null model: use X_calib (pure null data) when available, else fall
    # back to X.
    null_fit_data = X_calib if X_calib is not None else X
    null_model_fitted = clone(null_model).fit(null_fit_data)

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
        X_null = null_model_fitted.sample(n_samples, rng)
        scorer_null = clone(scorer_fitted).fit(X_null)
        null_ratios[b] = _max_score_ratio(scorer_null, X_null, specs, base_penalty_arr)

    # Calibrated multiplier: (1-level) quantile.
    c_star = float(np.quantile(null_ratios, 1.0 - level))

    # Return same type/shape as base_penalty.
    if np.ndim(base_penalty) == 0 or (
        isinstance(base_penalty, np.ndarray) and base_penalty.ndim == 0
    ):
        return float(c_star * float(base_penalty))

    result = c_star * base_penalty_arr
    if isinstance(base_penalty, np.ndarray):
        return result
    return float(result[0])
