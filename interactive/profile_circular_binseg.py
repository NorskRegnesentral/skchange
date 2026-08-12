"""Profile CircularBinarySegmentation on a simple univariate Gaussian series.

Runs CircularBinarySegmentation with n_samples = 10_000 using the default
L2TransientScore and a CostTransientScore wrapping L1Cost, and prints the
cProfile stats sorted by cumulative time. A warm-up run is performed first so
that Numba JIT compilation is not counted in the profile.
"""

import cProfile
import pstats
import time

import numpy as np

from skchange.new_api.detectors import CircularBinarySegmentation
from skchange.new_api.interval_scorers import (
    CostTransientScore,
    L1Cost,
    L2TransientScore,
)

N_SAMPLES = 10_000
SEED = 0
TOP_N = 20


def make_data(n_samples: int, seed: int) -> np.ndarray:
    """Simple univariate Gaussian series with zero changepoints."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n_samples).reshape(-1, 1)


def profile_circular_binseg(transient_score, X: np.ndarray, label: str) -> None:
    # Warm-up run so Numba JIT compilation is not measured.
    CircularBinarySegmentation(transient_score=transient_score).fit_predict(X)

    detector = CircularBinarySegmentation(transient_score=transient_score)
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
    print(
        f"Profiling CircularBinarySegmentation with L2TransientScore "
        f"on n_samples = {N_SAMPLES}"
    )
    print("=" * 80)
    profile_circular_binseg(L2TransientScore(), X, "L2TransientScore")

    print("=" * 80)
    print(
        f"Profiling CircularBinarySegmentation with CostTransientScore(L1Cost) "
        f"on n_samples = {N_SAMPLES}"
    )
    print("=" * 80)
    profile_circular_binseg(
        CostTransientScore(L1Cost()), X, "CostTransientScore(L1Cost)"
    )
