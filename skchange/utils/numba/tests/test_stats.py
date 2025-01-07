import numpy as np
import scipy.stats as st

from skchange.utils.numba.stats import kurtosis


def test_numba_kurtosis():
    seed = 523
    n_samples = 100
    p = 5
    t_dof = 5.0

    np.random.seed(seed)

    # Spd covariance matrix:
    random_nudge = np.random.randn(p).reshape(-1, 1)
    cov = np.eye(p) + 0.5 * random_nudge @ random_nudge.T

    mean = np.arange(p) * (-1 * np.ones(p)).cumprod()

    mv_t_dist = st.multivariate_t(loc=mean, shape=cov, df=t_dof)
    mv_t_samples = mv_t_dist.rvs(n_samples)

    sample_medians = np.median(mv_t_samples, axis=0)
    centered_samples = mv_t_samples - sample_medians

    # Test numba kurtosis:
    numba_kurtosis_val = kurtosis(centered_samples)
    scipy_kurtosis_val = st.kurtosis(centered_samples, fisher=True, bias=True, axis=0)

    assert np.all(np.isfinite(numba_kurtosis_val)), "Numba kurtosis should be finite."
    assert np.all(np.isfinite(scipy_kurtosis_val)), "Scipy kurtosis should be finite."
    (
        np.testing.assert_allclose(numba_kurtosis_val, scipy_kurtosis_val),
        "Kurtosis is off.",
    )
