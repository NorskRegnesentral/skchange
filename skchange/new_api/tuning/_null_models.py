"""Null-distribution samplers for FWER calibration.

A *null sampler* produces change-free ("null") data sets that mimic the
distribution a detector would see when there is no changepoint. Calibration
draws many such null samples, runs the detector on each, and derives a penalty
scale that controls the family-wise error rate.

Two sampler contracts are supported:

- :class:`BaseDataSampler` -- *data-based*. Fitted on observed change-free data
  and resamples from it. ``sample(n_samples, rng)``.
- :class:`BaseParametricSampler` -- *parametric*. Draws from a fixed
  distribution; never fitted. ``sample(n_samples, n_features, rng)``.

A plain callable ``f(n_samples, n_features, rng) -> np.ndarray`` is also accepted
anywhere a sampler is expected, as an escape hatch for custom null models.
"""

import numpy as np
from sklearn.base import BaseEstimator, clone

from skchange.new_api.utils.validation import check_is_fitted


class BaseDataSampler(BaseEstimator):
    """Base class for data-based null samplers.

    Subclasses implement :meth:`sample`, drawing ``n_samples`` rows that imitate
    the change-free data passed to :meth:`fit`.
    """

    def fit(self, X, y=None) -> "BaseDataSampler":
        """Store the change-free data to resample from.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Change-free data the null samples should imitate.
        y : Ignored
            Present for API consistency.

        Returns
        -------
        self : BaseDataSampler
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"`X` must be 2-D, got shape {X.shape}.")
        self.data_ = X
        self.n_features_in_ = X.shape[1]
        return self

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a null sample of shape ``(n_samples, n_features_in_)``."""
        raise NotImplementedError("Subclasses must implement `sample`.")


class BaseParametricSampler(BaseEstimator):
    """Base class for parametric null samplers.

    Parametric samplers carry a fully specified distribution and are never
    fitted. Subclasses implement :meth:`sample`.
    """

    def sample(
        self, n_samples: int, n_features: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw a null sample of shape ``(n_samples, n_features)``."""
        raise NotImplementedError("Subclasses must implement `sample`.")


class PermutationSampler(BaseDataSampler):
    """Non-parametric null sampler that resamples rows of observed data.

    Resampling whole rows preserves the cross-feature (joint) distribution while
    destroying any temporal structure -- the "independent and identically
    distributed in time" null hypothesis. This is the default null model.

    Parameters
    ----------
    replace : bool, default=False
        If ``False``, draw rows without replacement (a permutation / subsample);
        this requires ``n_samples <= n_stored``. If ``True``, draw with
        replacement (a row bootstrap), which works for any ``n_samples``.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.tuning import PermutationSampler
    >>> rng = np.random.default_rng(0)
    >>> sampler = PermutationSampler().fit(rng.normal(size=(100, 2)))
    >>> sampler.sample(50, rng).shape
    (50, 2)
    """

    def __init__(self, replace: bool = False):
        self.replace = replace

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        check_is_fitted(self, "data_")
        n_stored = self.data_.shape[0]
        if not self.replace and n_samples > n_stored:
            raise ValueError(
                f"PermutationSampler with replace=False cannot draw "
                f"{n_samples} rows from only {n_stored} stored rows. "
                f"Use replace=True or provide more null data."
            )
        idx = rng.choice(n_stored, size=n_samples, replace=self.replace)
        return self.data_[idx].astype(np.float64, copy=True)


class GaussianMCSampler(BaseParametricSampler):
    """Parametric null sampler drawing i.i.d. Gaussian data.

    Each entry is drawn independently from ``N(mean, std**2)``. Use this when the
    change-free distribution is (approximately) known to be standard Gaussian; it
    gives the tightest calibration in that case.

    Parameters
    ----------
    mean : float, default=0.0
        Mean of the Gaussian.
    std : float, default=1.0
        Standard deviation of the Gaussian.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.tuning import GaussianMCSampler
    >>> rng = np.random.default_rng(0)
    >>> GaussianMCSampler().sample(50, 2, rng).shape
    (50, 2)
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def sample(
        self, n_samples: int, n_features: int, rng: np.random.Generator
    ) -> np.ndarray:
        return rng.normal(
            loc=self.mean, scale=self.std, size=(n_samples, n_features)
        ).astype(np.float64, copy=False)


_NAMED_SAMPLERS = {
    "permutation": PermutationSampler,
    "gaussian": GaussianMCSampler,
}


def _resolve_sampler(sampler):
    """Resolve a sampler spec to a sampler instance or callable.

    Accepts a string alias (``"permutation"``, ``"gaussian"``), a sampler
    instance, or a plain callable ``f(n_samples, n_features, rng) -> ndarray``.
    """
    if isinstance(sampler, str):
        if sampler not in _NAMED_SAMPLERS:
            raise ValueError(
                f"Unknown sampler {sampler!r}. "
                f"Choose one of {sorted(_NAMED_SAMPLERS)} or pass a sampler "
                f"instance / callable."
            )
        return _NAMED_SAMPLERS[sampler]()
    if isinstance(sampler, (BaseDataSampler, BaseParametricSampler)):
        return sampler
    if callable(sampler):
        return sampler
    raise TypeError(
        "`sampler` must be a string alias, a BaseDataSampler / "
        "BaseParametricSampler instance, or a callable; "
        f"got {type(sampler).__name__!r}."
    )


def make_null_draw(sampler, X, X_calib, n_samples: int, n_features: int):
    """Build a ``draw(rng) -> ndarray`` closure for one null sample.

    Data-based samplers are fitted on ``X_calib`` if given, otherwise ``X``.
    Parametric samplers and callables ignore both and draw from their own
    distribution. The returned closure always yields arrays of shape
    ``(n_samples, n_features)``.
    """
    resolved = _resolve_sampler(sampler)
    null_source = X_calib if X_calib is not None else X

    if isinstance(resolved, BaseDataSampler):
        fitted = clone(resolved).fit(null_source)
        if fitted.n_features_in_ != n_features:
            raise ValueError(
                f"Null data has {fitted.n_features_in_} features but the "
                f"detection data has {n_features}. They must match."
            )
        return lambda rng: fitted.sample(n_samples, rng)

    if isinstance(resolved, BaseParametricSampler):
        return lambda rng: resolved.sample(n_samples, n_features, rng)

    # Plain callable escape hatch.
    return lambda rng: np.asarray(
        resolved(n_samples, n_features, rng), dtype=np.float64
    )
