"""Base class for penalties and penalty functions."""

import numpy as np
from skbase.base import BaseObject


class BasePenalty(BaseObject):
    """Base class template for penalties.

    This is a common base class for penalties in skchange. It is used as an internal
    building block for most detectors in skchange. The class provides a common interface
    to set and get both base and scaled penalties or penalty functions.

    Penalties are used to penalize the number of change points or segments in a change
    detection problem to avoid overfitting.

    The simplest type of penalty is a constant penalty value per additional change point
    or segment. For multivariate change detection, some algorithms are adaptive to
    the sparsity of the change, i.e., how many variables are affected by the change.
    In this case, the penalty is an increasing function of the sparsity of the change.
    """

    # Type of penalty:
    #  1. "constant": a penalty that is constant in the sparsity of the change and per
    #     additional change point/segment.
    #  2. "linear": a penalty that is linear in the sparsity of the change and therefore
    #     not constant per additional change point/segment.
    #  3. "nonlinear": a penalty that is nonlinear in the sparsity of the change and
    #     therefore not constant per additional change point/segment.
    penalty_type = None

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        super().__init__()

        if self.scale <= 0:
            raise ValueError("scale must be positive")

    @property
    def base_values(self) -> np.ndarray:
        """Get the base penalty values.

        Returns
        -------
        base_values : np.ndarray
            1D array of base (unscaled) penalty values. The shape of the output depends
            on the `penalty_type`:

            * If ``"constant"``, the output is of shape ``(1,)``.
            * If ``"linear"`` or ``"nonlinear"``, the output is of shape ``(p,)``,
            where ``p`` is the number of variables/columns in the data being analysed.
            Element ``i`` of the array is the base penalty value for ``i+1`` variables
            being affected by the change. The base penalty vector is non-decreasing.
        """
        raise NotImplementedError("abstract method")

    @property
    def values(self) -> np.ndarray:
        """Get the penalty values.

        Returns
        -------
        values : np.ndarray
            1D array of penalty values given by `scale * base_values`. The shape of the
            output depends on the `penalty_type`:

            * If ``"constant"``, the output is of shape ``(1,)``.
            * If ``"linear"`` or ``"nonlinear"``, the output is of shape ``(p,)``,
            where ``p`` is the number of variables/columns in the data being analysed.
            Element ``i`` of the array is the penalty value for ``i+1`` variables
            being affected by the change. The penalty vector is non-decreasing.
        """
        return self.scale * self.base_values


class ConstantPenalty(BasePenalty):
    """Constant penalty."""

    penalty_type = "constant"

    def __init__(self, value: float, scale: float = 1.0):
        super().__init__(scale)
        self.value = value

    @property
    def base_values(self) -> np.ndarray:
        """Get the base penalty values.

        Returns
        -------
        base_values : np.ndarray
            1D array of base (unscaled) penalty values. The shape of the output depends
            on the `penalty_type`:

            * If ``"constant"``, the output is of shape ``(1,)``.
            * If ``"linear"`` or ``"nonlinear"``, the output is of shape ``(p,)``,
            where ``p`` is the number of variables/columns in the data being analysed.
        """
        return np.array([self.value])


class BICPenalty(BasePenalty):
    """Bayesian Information Criterion (BIC) penalty.

    The BIC penalty is a constant penalty given by ``(n_params + 1) * log(n)``, where
    `n` is the sample size and `n_params` is the number of parameters per segment in the
    model across all variables. The ``+ 1`` term accounts for the change point
    parameter.

    Parameters
    ----------
    n : int
        Sample size.
    n_params : int
        Number of parameters per segment in the model across all variables.
    scale : float, optional, default=1.0
        Scaling factor for the penalty.
    """

    penalty_type = "constant"

    def __init__(self, n: int, n_params: int, scale: float = 1.0):
        super().__init__(scale)
        self.n = n
        self.n_params = n_params

    @property
    def base_values(self) -> np.ndarray:
        """Get the base penalty values.

        Returns
        -------
        base_values : np.ndarray
            Shape ``(1,)`` array with the base (unscaled) penalty values.
        """
        # +1 due to the additional change point parameter in the model.
        base_penalty = (self.n_params + 1) * np.log(self.n)
        return np.array([base_penalty])
