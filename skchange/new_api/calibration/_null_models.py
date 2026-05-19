"""Null model implementations for calibration."""

from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class BaseNullModel(BaseEstimator):
    """Base class for null models used in calibration.

    Subclasses must implement :meth:`fit` and :meth:`sample`.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y=None) -> "BaseNullModel":
        """Fit the null model to training data."""

    @abstractmethod
    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a null sample of shape (n_samples, n_features_in_)."""


class PermutationNullModel(BaseNullModel):
    """Null model that resamples rows of the training data.

    Parameters
    ----------
    replace : bool, default=False
        If False (default), perform a strict permutation (rows shuffled without
        replacement — exact marginal distributions). If True, perform a
        non-parametric row bootstrap (sampling with replacement).
    """

    def __init__(self, replace: bool = False):
        self.replace = replace

    def fit(self, X: np.ndarray, y=None) -> "PermutationNullModel":
        """Store training data for resampling."""
        self.X_ = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = self.X_.shape[1]
        return self

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a permutation/bootstrap sample.

        With ``replace=False``: rows of X are sampled without replacement
        (a row-permutation when ``n_samples == n_train``).
        With ``replace=True``: rows are resampled with replacement (bootstrap).
        """
        check_is_fitted(self, "X_")
        n_train = self.X_.shape[0]
        if self.replace:
            idx = rng.integers(0, n_train, size=n_samples)
        else:
            idx = rng.choice(n_train, size=n_samples, replace=False)
        return self.X_[idx].copy()


class BlockBootstrapNullModel(BaseNullModel):
    """Circular block bootstrap null model.

    Parameters
    ----------
    block_length : int, default=10
        Length of each bootstrap block.
    """

    def __init__(self, block_length: int = 10):
        self.block_length = block_length

    def fit(self, X: np.ndarray, y=None) -> "BlockBootstrapNullModel":
        """Store training data for block resampling."""
        self.X_ = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = self.X_.shape[1]
        return self

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a circular block bootstrap sample."""
        check_is_fitted(self, "X_")
        n_train = self.X_.shape[0]
        bl = self.block_length
        n_blocks = int(np.ceil(n_samples / bl))
        starts = rng.integers(0, n_train, size=n_blocks)
        rows = []
        for s in starts:
            for k in range(bl):
                rows.append((s + k) % n_train)
        idx = np.array(rows[:n_samples], dtype=np.intp)
        return self.X_[idx].copy()


class GaussianNullModel(BaseNullModel):
    """Null model that fits an i.i.d. Gaussian per feature.

    Fits the mean and standard deviation from training data and draws
    independent Gaussian samples.
    """

    def fit(self, X: np.ndarray, y=None) -> "GaussianNullModel":
        """Fit mean and std per feature."""
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=1)
        self.n_features_in_ = X.shape[1]
        return self

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Draw i.i.d. Gaussian samples."""
        check_is_fitted(self, "mean_")
        z = rng.standard_normal(size=(n_samples, self.n_features_in_))
        return (z * self.std_ + self.mean_).astype(np.float64)


class MCNullModel(BaseNullModel):
    """Null model driven by a user-supplied data-generating process (DGP).

    Parameters
    ----------
    dgp : callable
        A callable with signature ``dgp(n_samples, n_features, rng) -> ndarray``
        that returns an array of shape ``(n_samples, n_features)``.
    """

    def __init__(self, dgp):
        self.dgp = dgp

    def fit(self, X: np.ndarray, y=None) -> "MCNullModel":
        """Record data dimensions for use in sample()."""
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        return self

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a sample using the user-supplied DGP."""
        check_is_fitted(self, "n_features_in_")
        return np.asarray(
            self.dgp(n_samples, self.n_features_in_, rng), dtype=np.float64
        )
