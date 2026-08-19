"""Benchmark FPOP vs PELT vs MovingWindow and sanity-check outputs on three scenarios.

Scenarios
---------
1. No changepoints  – stationary Gaussian signal.
2. One changepoint  – single mean shift at the midpoint.
3. Many changepoints – alternating means every N_SEGMENT samples.

Timing uses ``timeit`` after a warm-up run so Numba JIT is not counted.
"""

import timeit

import numpy as np

from skchange.detectors import FPOP, MovingWindow, PELT

SEED = 0
N_REPEATS = 5  # timeit repetitions per scenario/detector


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def make_no_cpt(n: int, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    X = rng.normal(0, 1, (n, 1))
    return X, []


def make_one_cpt(n: int, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    mid = n // 2
    X = np.concatenate([rng.normal(0, 1, (mid, 1)), rng.normal(5, 1, (n - mid, 1))])
    return X, [mid]


def make_many_cpts(
    n: int, seg_len: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[int]]:
    means = [0, 5, -3, 8, 2, -5, 6, -2, 4, -4]
    segments = []
    true_cpts = []
    pos = 0
    for i, mu in enumerate(means):
        length = seg_len if i < len(means) - 1 else n - pos
        segments.append(rng.normal(mu, 1, (length, 1)))
        pos += length
        if pos < n:
            true_cpts.append(pos)
    return np.concatenate(segments), true_cpts


# ---------------------------------------------------------------------------
# Sanity check helpers
# ---------------------------------------------------------------------------


def _cpts_close(detected: np.ndarray, expected: list[int], tol: int) -> bool:
    if len(detected) != len(expected):
        return False
    return all(
        abs(int(d) - e) <= tol for d, e in zip(sorted(detected), sorted(expected))
    )


def run_sanity_checks(n: int = 1_000, seg_len: int = 100, tol: int = 10) -> None:
    rng = np.random.default_rng(SEED)
    scenarios = [
        ("no changepoints", *make_no_cpt(n, rng)),
        ("one changepoint", *make_one_cpt(n, rng)),
        ("many changepoints", *make_many_cpts(n, seg_len, rng)),
    ]

    print("=" * 70)
    print("Sanity checks")
    print("=" * 70)

    for label, X, expected in scenarios:
        fpop_cpts = FPOP().fit(X).predict(X)
        pelt_cpts = PELT().fit(X).predict(X)
        mw_cpts = MovingWindow().fit(X).predict(X)

        fpop_ok = _cpts_close(fpop_cpts, expected, tol)
        pelt_ok = _cpts_close(pelt_cpts, expected, tol)
        mw_ok = _cpts_close(mw_cpts, expected, tol)

        print(f"\n  Scenario    : {label}")
        print(f"  Expected    : {expected}")
        print(
            f"  FPOP        : {fpop_cpts.tolist()}  {'OK' if fpop_ok else 'MISMATCH'}"
        )
        print(
            f"  PELT        : {pelt_cpts.tolist()}  {'OK' if pelt_ok else 'MISMATCH'}"
        )
        print(f"  MovingWindow: {mw_cpts.tolist()}  {'OK' if mw_ok else 'MISMATCH'}")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _warm_up(X: np.ndarray) -> None:
    FPOP().fit(X).predict(X)
    PELT().fit(X).predict(X)
    MovingWindow().fit(X).predict(X)


def run_benchmark(sizes: list[int], seg_len: int = 100) -> None:
    print("\n" + "=" * 70)
    print(f"Benchmark  (n_repeats={N_REPEATS} per size/detector)")
    print("=" * 70)
    print(
        f"  {'n':>8}  {'FPOP (s)':>12}  {'PELT (s)':>12}  {'MW (s)':>12}  {'PELT/FPOP':>10}  {'MW/FPOP':>9}"
    )
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 10}  {'-' * 9}")

    rng = np.random.default_rng(SEED)

    for n in sizes:
        X, _ = make_many_cpts(n, seg_len, rng)

        # Warm up all detectors so JIT is excluded.
        _warm_up(X[:100].reshape(-1, 1) if n > 100 else X)

        fpop_time = (
            timeit.timeit(lambda: FPOP().fit(X).predict(X), number=N_REPEATS)
            / N_REPEATS
        )
        pelt_time = (
            timeit.timeit(lambda: PELT().fit(X).predict(X), number=N_REPEATS)
            / N_REPEATS
        )
        mw_time = (
            timeit.timeit(lambda: MovingWindow().fit(X).predict(X), number=N_REPEATS)
            / N_REPEATS
        )

        print(
            f"  {n:>8,}  {fpop_time:>12.4f}  {pelt_time:>12.4f}"
            f"  {mw_time:>12.4f}  {pelt_time / fpop_time:>9.2f}x"
            f"  {mw_time / fpop_time:>8.2f}x"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_sanity_checks(n=1_000, seg_len=100, tol=10)
    run_benchmark(sizes=[1_000, 5_000, 10_000, 50_000], seg_len=100)
