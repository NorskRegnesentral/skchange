"""Null sampler implementations for MC-based penalty calibration.

There are two conceptually distinct types of null samplers:

* **Data-based samplers** (:class:`BaseDataSampler`): generate null samples by
  resampling rows of observed training data.  They require a :meth:`fit` call
  before :meth:`sample`.  Use these when you have a representative null series
  (no changes) available.

* **Parametric / MC samplers** (:class:`BaseParametricSampler`): generate null
  samples from a fully specified parametric distribution.  They require *no*
  training data — distribution parameters are fixed at construction time.  Use
  these when you want to calibrate purely via Monte Carlo simulation without
  needing observed null data.
"""

from abc import abstractmethod
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class BaseDataSampler(BaseEstimator):
    """Abstract base class for data-based null samplers.

    Subclasses generate null samples by resampling rows of observed training
    data.  They require a call to :meth:`fit` before :meth:`sample`.

    Subclasses must implement :meth:`fit` and :meth:`sample`.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y=None) -> "BaseDataSampler":
        """Fit the sampler by storing training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Observed null series (no changes expected).
        y : None
            Ignored.

        Returns
        -------
        self
        """

    @abstractmethod
    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a null sample.

        Parameters
        ----------
        n_samples : int
            Number of rows to generate.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        ndarray of shape (n_samples, n_features_in_)
        """


class BaseParametricSampler(BaseEstimator):
    """Abstract base class for parametric (MC) null samplers.

    Subclasses generate null samples from a fully specified parametric
    distribution.  No training data is required — all parameters are fixed
    at construction time.

    Subclasses must implement :meth:`sample`.
    """

    @abstractmethod
    def sample(
        self, n_samples: int, n_features: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw a null sample.

        Parameters
        ----------
        n_samples : int
            Number of rows to generate.
        n_features : int
            Number of features (columns) to generate.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
        """


# ---------------------------------------------------------------------------
# Data-based samplers
# ---------------------------------------------------------------------------


class PermutationSampler(BaseDataSampler):
    """Null sampler that resamples rows of the training data.

    Parameters
    ----------
    replace : bool, default=False
        If ``False`` (default), rows are sampled without replacement (a strict
        row-permutation when ``n_samples == n_train``).  If ``True``, rows are
        resampled with replacement (non-parametric row bootstrap).
    """

    def __init__(self, replace: bool = False):
        self.replace = replace

    def fit(self, X: np.ndarray, y=None) -> "PermutationSampler":
        """Store training data for resampling."""
        self.X_ = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = self.X_.shape[1]
        return self

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a permutation or bootstrap sample of the training rows."""
        check_is_fitted(self, "X_")
        n_train = self.X_.shape[0]
        if not self.replace and n_samples > n_train:
            raise ValueError(
                f"PermutationSampler with replace=False cannot draw {n_samples} "
                f"samples from {n_train} training rows without replacement. "
                f"Use replace=True or provide X_train with at least {n_samples} rows."
            )
        if self.replace:
            idx = rng.integers(0, n_train, size=n_samples)
        else:
            idx = rng.choice(n_train, size=n_samples, replace=False)
        return self.X_[idx].copy()


class BlockBootstrapSampler(BaseDataSampler):
    """Circular block bootstrap null sampler.

    Preserves short-range temporal dependence by resampling contiguous blocks
    of rows from the training data with circular wrap-around.

    Parameters
    ----------
    block_length : int, default=10
        Length of each bootstrap block.
    """

    def __init__(self, block_length: int = 10):
        self.block_length = block_length

    def fit(self, X: np.ndarray, y=None) -> "BlockBootstrapSampler":
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


# ---------------------------------------------------------------------------
# Parametric / MC samplers
# ---------------------------------------------------------------------------


class GaussianMCSampler(BaseParametricSampler):
    """Parametric null sampler drawing i.i.d. Gaussian observations.

    All parameters are fixed at construction time — no training data is
    required.  Each feature is drawn independently from
    ``Normal(mean, std**2)``.

    Parameters
    ----------
    mean : float, default=0.0
        Mean of the Gaussian distribution.
    std : float, default=1.0
        Standard deviation of the Gaussian distribution.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def sample(
        self, n_samples: int, n_features: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw i.i.d. Gaussian samples of shape (n_samples, n_features)."""
        return rng.normal(self.mean, self.std, size=(n_samples, n_features))


class MCSimulator(BaseParametricSampler):
    """Parametric null sampler driven by a user-supplied data-generating process.

    All parameters are fixed at construction time — no training data is
    required.

    Parameters
    ----------
    dgp : callable
        A callable with signature
        ``dgp(n_samples: int, n_features: int, rng: Generator) -> ndarray``
        that returns an array of shape ``(n_samples, n_features)``.
    """

    def __init__(self, dgp: Callable[[int, int, np.random.Generator], np.ndarray]):
        self.dgp = dgp

    def sample(
        self, n_samples: int, n_features: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw a sample using the user-supplied DGP."""
        return np.asarray(self.dgp(n_samples, n_features, rng), dtype=np.float64)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_SAMPLER_ALIASES: dict[str, type] = {
    "permutation": PermutationSampler,
    "block_bootstrap": BlockBootstrapSampler,
    "gaussian": GaussianMCSampler,
}


def _resolve_sampler(
    resampling: "str | BaseDataSampler | BaseParametricSampler",
) -> "BaseDataSampler | BaseParametricSampler":
    """Resolve a string alias or sampler instance to a sampler object.

    Parameters
    ----------
    resampling : str or sampler instance
        Either a string alias (``"permutation"``, ``"block_bootstrap"``,
        ``"gaussian"``) or an already-constructed
        :class:`BaseDataSampler` / :class:`BaseParametricSampler` instance.

    Returns
    -------
    BaseDataSampler or BaseParametricSampler
        A sampler instance (with default parameters when constructed from a
        string alias).

    Raises
    ------
    ValueError
        If ``resampling`` is a string that is not a recognised alias.
    TypeError
        If ``resampling`` is neither a string nor a sampler instance.
    """
    if isinstance(resampling, (BaseDataSampler, BaseParametricSampler)):
        return resampling
    if isinstance(resampling, str):
        if resampling not in _SAMPLER_ALIASES:
            raise ValueError(
                f"Unknown resampling alias {resampling!r}. "
                f"Valid aliases are: {sorted(_SAMPLER_ALIASES)}."
            )
        return _SAMPLER_ALIASES[resampling]()
    raise TypeError(
        f"'resampling' must be a string alias or a BaseDataSampler / "
        f"BaseParametricSampler instance, got {type(resampling).__name__!r}."
    )
