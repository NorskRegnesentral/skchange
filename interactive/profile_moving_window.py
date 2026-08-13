"""Profile MovingWindow on a simple univariate Gaussian series.

Runs MovingWindow with n_samples = 100_000 using both the default CUSUM change
score and a CostChangeScore wrapping L1Cost, and prints the cProfile stats
sorted by cumulative time. A warm-up run is performed first so that Numba JIT
compilation is not counted in the profile.
"""

import cProfile
import pstats
import time

import numpy as np

from skchange.detectors import MovingWindow
from skchange.interval_scorers import CUSUM, CostChangeScore, L1Cost

N_SAMPLES = 10_000
SEED = 0
TOP_N = 20


def make_data(n_samples: int, seed: int) -> np.ndarray:
    """Simple univariate Gaussian series with zero changepoints."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n_samples).reshape(-1, 1)


def profile_moving_window(change_score, X: np.ndarray, label: str) -> None:
    # Warm-up run so Numba JIT compilation is not measured.
    MovingWindow(change_score=change_score).fit_predict(X)

    detector = MovingWindow(change_score=change_score)
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
    print(f"Profiling MovingWindow with CUSUM on n_samples = {N_SAMPLES}")
    print("=" * 80)
    profile_moving_window(CUSUM(), X, "CUSUM")

    print("=" * 80)
    print(
        f"Profiling MovingWindow with CostChangeScore(L1Cost) "
        f"on n_samples = {N_SAMPLES}"
    )
    print("=" * 80)
    profile_moving_window(CostChangeScore(L1Cost()), X, "CostChangeScore(L1Cost)")
