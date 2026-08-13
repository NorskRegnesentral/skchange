"""Profile CAPA on a simple univariate Gaussian series.

Runs CAPA with n_samples = 10_000 using both the default L2Saving and L1Saving,
and prints the cProfile stats sorted by cumulative time. A warm-up run is
performed first so that Numba JIT compilation is not counted in the profile.
"""

import cProfile
import pstats
import time

import numpy as np

from skchange.detectors import CAPA
from skchange.interval_scorers import L1Saving, L2Saving

N_SAMPLES = 10_000
SEED = 0
TOP_N = 20


def make_data(n_samples: int, seed: int) -> np.ndarray:
    """Simple univariate Gaussian series with zero anomalies."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n_samples).reshape(-1, 1)


def profile_capa(saving, X: np.ndarray, label: str) -> None:
    # Warm-up run so Numba JIT compilation is not measured.
    CAPA(segment_saving=saving).fit_predict(X)

    detector = CAPA(segment_saving=saving)
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    cpts = detector.fit_predict(X)
    profiler.disable()
    elapsed = time.perf_counter() - t0
    print(f"[{label}] elapsed (incl. profiler overhead): {elapsed:.3f} s")
    print(f"[{label}] detected {len(cpts)} anomalies")
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(TOP_N)


if __name__ == "__main__":
    X = make_data(N_SAMPLES, SEED)

    print("=" * 80)
    print(f"Profiling CAPA with L2Saving on n_samples = {N_SAMPLES}")
    print("=" * 80)
    profile_capa(L2Saving(), X, "L2Saving")

    print("=" * 80)
    print(f"Profiling CAPA with L1Saving on n_samples = {N_SAMPLES}")
    print("=" * 80)
    profile_capa(L1Saving(), X, "L1Saving")
