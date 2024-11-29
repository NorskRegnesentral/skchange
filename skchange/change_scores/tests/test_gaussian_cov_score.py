import numpy as np
from scipy.special import digamma

from skchange.change_scores.gaussian_cov_score import (
    GaussianCovScore,
    half_integer_digamma,
)


def test_digamma():
    integer_vals = np.arange(1, 25)

    integer_vals = np.concatenate([integer_vals, 100 + integer_vals])

    manual_digamma = np.array(list(map(half_integer_digamma, 2 * integer_vals)))
    scipy_digamma = digamma(integer_vals)

    np.testing.assert_allclose(manual_digamma, scipy_digamma)


def test_GaussianCovScore():
    np.random.seed(0)
    X_1 = np.random.normal(size=(100, 3), loc=[1.0, -0.2, 0.5], scale=[1.0, 0.5, 1.5])
    X_2 = np.random.normal(size=(100, 3), loc=[-1.0, 0.2, -0.5], scale=[4.0, 1.5, 2.8])

    X = np.concatenate([X_1, X_2, X_1], axis=0)
    cuts = np.array([[0, 50, 100], [0, 100, 200], [100, 200, 300], [0, 150, 300]])

    scores = GaussianCovScore().fit(X).evaluate(cuts)

    assert scores.shape == (cuts.shape[0], 1)
    assert np.all(scores >= 0)
