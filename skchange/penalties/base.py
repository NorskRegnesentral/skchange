"""Base class for penalties and penalty functions."""

import numpy as np
import pandas as pd
from sktime.base import BaseEstimator

from ..base import BaseIntervalScorer
from ..utils.validation.data import as_2d_array


class BasePenalty(BaseEstimator):
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

        if self.scale < 0:
            raise ValueError("scale must be non-negative")

    def fit(
        self, X: pd.DataFrame | pd.Series | np.ndarray, scorer: BaseIntervalScorer
    ) -> "BasePenalty":
        """Fit the penalty to data and a scorer.

        The default implementation gets the number of samples and variables in the data
        and sets the `_is_fitted` attribute to `True`. Subclasses should implement the
        `_fit` method if more attributes of the data are needed to fit the penalty.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            The data to fit the penalty to.
        scorer : BaseIntervalScorer
            The interval scorer to fit the penalty to.

        Returns
        -------
        self
            Reference to a fitted instance of self.
        """
        X = as_2d_array(X)
        self.n_ = X.shape[0]
        self.p_ = X.shape[1]
        self.n_params_total_ = scorer.get_param_size(self.p_)
        self.n_params_per_variable_ = scorer.get_param_size(1)

        self._fit(X=X, scorer=scorer)

        self._is_fitted = True
        return self

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
        if not self.is_fitted:
            raise ValueError("The penalty must be fitted before getting values.")

        base_values = np.atleast_1d(self._base_values)
        return self.scale * base_values

    @property
    def _base_values(self) -> np.ndarray | float:
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

    def _fit(
        self, X: pd.DataFrame | pd.Series | np.ndarray, scorer: BaseIntervalScorer
    ) -> "BasePenalty":
        """Fit the penalty to data and a scorer.

        This method should be implemented if more fitting is needed than just obtaining
        the number of samples and variables in the data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            The data to fit the penalty to.
        scorer : BaseIntervalScorer
            The interval scorer to fit the penalty to.

        Returns
        -------
        self
            Reference to self.
        """
        return self
