"""Profile SeededBinarySegmentation on a simple univariate Gaussian series.

Runs SeededBinarySegmentation with n_samples = 10_000 using both the default
CUSUM change score and a CostChangeScore wrapping L1Cost, and prints the
cProfile stats sorted by cumulative time. A warm-up run is performed first
so that Numba JIT compilation is not counted in the profile.
"""

import cProfile
import pstats
import time

import numpy as np

from skchange.new_api.detectors import SeededBinarySegmentation
from skchange.new_api.interval_scorers import CUSUM, CostChangeScore, L1Cost

N_SAMPLES = 100_000
SEED = 0
TOP_N = 20


def make_data(n_samples: int, seed: int) -> np.ndarray:
    """Simple univariate Gaussian series with zero changepoints."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n_samples).reshape(-1, 1)


def profile_seeded_binseg(change_score, X: np.ndarray, label: str) -> None:
    # Warm-up run so Numba JIT compilation is not measured.
    SeededBinarySegmentation(change_score=change_score).fit_predict(X)

    detector = SeededBinarySegmentation(change_score=change_score)
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    cpts = detector.fit_predict(X)
    profiler.disable()
    elapsed = time.perf_counter() - t0
    print(f"[{label}] elapsed (incl. profiler overhead): {elapsed:.3f} s")
    print(f"[{label}] detected {len(cpts)} changepoints")
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(TOP_N)


if __name__ == "__main__":
    X = make_data(N_SAMPLES, SEED)

    print("=" * 80)
    print(f"Profiling SeededBinarySegmentation with CUSUM on n_samples = {N_SAMPLES}")
    print("=" * 80)
    profile_seeded_binseg(CUSUM(), X, "CUSUM")

    print("=" * 80)
    print(
        f"Profiling SeededBinarySegmentation with CostChangeScore(L1Cost) "
        f"on n_samples = {N_SAMPLES}"
    )
    print("=" * 80)
    profile_seeded_binseg(CostChangeScore(L1Cost()), X, "CostChangeScore(L1Cost)")
