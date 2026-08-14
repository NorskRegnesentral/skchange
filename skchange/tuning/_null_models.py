"""Null-distribution samplers for FWER calibration.

A *null sampler* produces change-free ("null") data sets that mimic the
distribution a detector would see when there is no changepoint. Calibration
draws many such null samples, runs the detector on each, and derives a penalty
scale that controls the family-wise error rate.

All samplers share one contract: ``sample(X, n_samples, rng) -> ndarray`` of
shape ``(n_samples, X.shape[1])``. Data-based samplers resample the rows of
``X``. Parametric samplers use only ``X.shape[1]`` to shape their draw and
ignore its values. A plain callable with the same signature is accepted
anywhere a sampler is expected, as an escape hatch for custom null models.
"""

import warnings

import numpy as np


class BaseNullSampler:
    """Base class for null-distribution samplers for FWER calibration.

    A null sampler produces change-free ("null") data sets used by FWER
    calibration. Subclasses are pure config objects that hold their
    hyperparameters on ``self`` and implement :meth:`sample`, which takes the
    reference data, a draw size, and a per-call RNG.

    Design notes
    ------------
    * **Stateless config object.** Only hyperparameters live on ``self``. No
      ``fit``, no cached data, no RNG. Samplers are cheap to pickle and safe
      to share across ``joblib`` workers.
    * **RNG per call.** The caller (e.g. ``CalibratedDetectorFWER``) owns the
      master seed, spawns child seeds via ``SeedSequence.spawn()``, and
      passes a fresh ``Generator`` to each ``sample`` call. This keeps
      parallel draws independent and individual draws reproducible.
    * **``X`` always required by the contract.** Parametric subclasses read
      only ``X.shape[1]``. Data-based subclasses resample its rows. One uniform
      signature, no optional arguments. Whether a caller must supply real
      reference data is advertised by :attr:`requires_reference_data`.
    * **Duck-typed contract.** Consumers only need
      ``sample(X, n_samples, rng) -> ndarray``. A plain callable with the
      same signature works in place of a subclass.

    Attributes
    ----------
    requires_reference_data : bool
        Whether the sampler needs real reference data to draw from. ``True``
        for data-based samplers (the default), ``False`` for parametric
        samplers that synthesise data from the feature count alone.
    """

    requires_reference_data: bool = True

    def sample(
        self,
        X: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw one null sample of shape ``(n_samples, X.shape[1])``.

        Parameters
        ----------
        X : ndarray of shape (n_ref, n_features)
            Reference data. Data-based samplers resample its rows.
            Parametric samplers use only its second dimension.
        n_samples : int
            Number of rows to return.
        rng : numpy.random.Generator
            Per-call random generator supplied by the caller.

        Returns
        -------
        ndarray of shape (n_samples, X.shape[1])
        """
        raise NotImplementedError("Subclasses must implement `sample`.")

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"


class PermutationSampler(BaseNullSampler):
    """Non-parametric null sampler that resamples rows of ``X``.

    Resampling whole rows preserves the cross-feature (joint) distribution while
    destroying any temporal structure. This is the "independent and identically
    distributed in time" null hypothesis, and the default null model.

    Parameters
    ----------
    replace : bool, default=False
        If ``False``, draw rows without replacement (a permutation or
        subsample), which requires ``n_samples <= len(X)``. If ``True``, draw
        with replacement (a row bootstrap), which works for any ``n_samples``.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.tuning import PermutationSampler
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 2))
    >>> PermutationSampler().sample(X, 50, rng).shape
    (50, 2)
    """

    def __init__(self, replace: bool = False):
        self.replace = replace

    def sample(
        self,
        X: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n_ref = X.shape[0]
        if not self.replace and n_samples > n_ref:
            raise ValueError(
                f"PermutationSampler with replace=False cannot draw "
                f"{n_samples} rows from only {n_ref} reference rows. "
                f"Use replace=True or provide more null data."
            )
        idx = rng.choice(n_ref, size=n_samples, replace=self.replace)
        return X[idx]


class GaussianSampler(BaseNullSampler):
    """Parametric null sampler drawing i.i.d. Gaussian data.

    Each entry is drawn independently from ``N(mean, std**2)``. Uses only
    ``X.shape[1]`` from the reference data, and the values are ignored. Use this
    when the change-free distribution is approximately known to be Gaussian. It
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
    >>> from skchange.tuning import GaussianSampler
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 2))
    >>> GaussianSampler().sample(X, 50, rng).shape
    (50, 2)
    """

    requires_reference_data: bool = False

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def sample(
        self,
        X: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return rng.normal(self.mean, self.std, (n_samples, X.shape[1]))


class BlockBootstrapSampler(BaseNullSampler):
    """Circular block bootstrap null sampler.

    Draws a null sample by copying contiguous blocks of ``X`` from random start
    positions, wrapping around the end of ``X`` (Politis and Romano, 1992).
    Blocks of length ``block_length`` are laid end to end until ``n_samples``
    rows are filled, with the final block truncated to fit. Whole rows are
    copied together, so dependence across features is preserved.

    Unlike :class:`PermutationSampler`, which reshuffles rows independently and
    destroys all temporal order, the block bootstrap keeps short-range temporal
    dependence inside each block. Use it when the change-free data is serially
    dependent (autocorrelated), where an i.i.d. permutation would misstate the
    false alarm rate.

    Parameters
    ----------
    block_length : int or None, default=None
        Length of each resampled block. ``None`` resolves at draw time to
        ``max(1, int(len(X) ** (1 / 3)))``, the cube-root rate for block
        bootstraps under weak dependence (Hall, Horowitz and Jing, 1995).
        ``block_length=1`` reduces to an i.i.d. row bootstrap.

    Notes
    -----
    A block bootstrap reproduces dependence only up to the block scale. At each
    join between two consecutive blocks the copied rows come from unrelated
    parts of ``X``, which adds a small artificial discontinuity. The circular
    scheme removes the edge under-weighting of the plain moving block bootstrap
    but not this join effect.

    References
    ----------
    Politis, D. N. and Romano, J. P. (1992). A circular block resampling
    procedure for stationary data.

    Hall, P., Horowitz, J. L. and Jing, B.-Y. (1995). On blocking rules for the
    bootstrap with dependent data. Biometrika, 82(3), 561-574.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.tuning import BlockBootstrapSampler
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 2))
    >>> BlockBootstrapSampler(block_length=10).sample(X, 50, rng).shape
    (50, 2)
    """

    def __init__(self, block_length: int | None = None):
        self.block_length = block_length

    def _effective_block_length(self, n_ref: int) -> int:
        """Resolve the block length for a reference pool of ``n_ref`` rows."""
        if self.block_length is None:
            return max(1, int(n_ref ** (1 / 3)))
        return self.block_length

    def sample(
        self,
        X: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n_ref = X.shape[0]
        block_length = self._effective_block_length(n_ref)
        if block_length < 1:
            raise ValueError(f"block_length must be at least 1, got {block_length}.")
        if n_ref < n_samples:
            warnings.warn(
                f"BlockBootstrapSampler drew {n_samples} rows from a reference "
                f"pool of only {n_ref} rows. The circular bootstrap wraps around, "
                f"but calibration may be unreliable with so few rows.",
                UserWarning,
                stacklevel=2,
            )

        out = np.empty((n_samples, X.shape[1]), dtype=np.float64)
        t = 0
        while t < n_samples:
            start = int(rng.integers(0, n_ref))
            length = min(block_length, n_samples - t)
            idx = np.arange(start, start + length) % n_ref
            out[t : t + length] = X[idx]
            t += length
        return out


_NAMED_SAMPLERS = {
    "permutation": PermutationSampler,
    "gaussian": GaussianSampler,
    "block_bootstrap": BlockBootstrapSampler,
}


def resolve_sampler(sampler):
    """Resolve a sampler spec to a callable ``(X, n_samples, rng) -> ndarray``.

    Accepts a string alias (``"permutation"``, ``"gaussian"``), an object with
    a ``sample`` method, or a plain callable with the same signature.
    """
    if isinstance(sampler, str):
        if sampler not in _NAMED_SAMPLERS:
            raise ValueError(
                f"Unknown sampler {sampler!r}. "
                f"Choose one of {sorted(_NAMED_SAMPLERS)} or pass a sampler "
                f"instance / callable."
            )
        sampler = _NAMED_SAMPLERS[sampler]()
    if hasattr(sampler, "sample"):
        return sampler.sample
    if callable(sampler):
        return sampler
    raise TypeError(
        "`sampler` must be a string alias, an object with a `sample` method, "
        f"or a callable. Got {type(sampler).__name__!r}."
    )


def sampler_requires_data(sampler) -> bool:
    """Whether a sampler needs real reference data (``X``) to draw from.

    Parametric samplers (e.g. ``GaussianSampler``) synthesise data from the
    feature count alone and return ``False``. Data-based samplers and plain
    callables (whose needs cannot be introspected) return ``True``, so callers
    must supply ``X`` to be safe.

    Parameters
    ----------
    sampler : str, sampler instance, or callable
        The same specification accepted by :func:`resolve_sampler`.

    Returns
    -------
    bool
        ``True`` if reference data is required, ``False`` otherwise.
    """
    if isinstance(sampler, str):
        cls = _NAMED_SAMPLERS.get(sampler)
        if cls is None:
            return True
        sampler = cls()
    return bool(getattr(sampler, "requires_reference_data", True))
