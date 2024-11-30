"""Direct L2 saving for a zero-valued baseline mean."""

__author__ = ["Tveten"]

import numpy as np
from numpy.typing import ArrayLike

from skchange.anomaly_scores.base import BaseSaving
from skchange.utils.numba import njit
from skchange.utils.numba.stats import col_cumsum
from skchange.utils.validation.data import as_2d_array


@njit
def l2_saving(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
) -> np.ndarray:
    """
    Calculate the L2 saving for a zero-valued baseline mean.

    Parameters
    ----------
    starts : `np.ndarray`
        Start indices of the segments.
    ends : `np.ndarray`
        End indices of the segments.
    sums : `np.ndarray`
        Cumulative sum of the input data, with a row of 0-entries as the first row.

    Returns
    -------
    savings : `np.ndarray`
        2D array of savings for each segment (rows) and component (columns).
    """
    n = (ends - starts).reshape(-1, 1)
    saving = (sums[ends] - sums[starts]) ** 2 / n
    return saving


class L2Saving(BaseSaving):
    """L2 saving for a zero-valued baseline mean.

    The L2 saving for a zero-mean can be computed more efficiently directly, rather
    than evaluating the cost separately for the baseline and optimized parameters.
    See [1]_ for details.

    Note that the data is assumed to have a zero-valued baseline
    mean for this saving to work, and that the data is assumed to be preprocessed
    accordingly. This can be achieved by subtracting the median of the input data,
    as it is a robust estimator of the baseline mean.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). A linear time method\
        for the detection of collective and point anomalies. Statistical Analysis and\
        DataMining: The ASA Data Science Journal, 15(4), 494-508.
    """

    def __init__(self):
        super().__init__()

    @property
    def min_size(self) -> int:
        """Minimum valid size of the interval to evaluate."""
        return 1

    def get_param_size(self, p: int) -> int:
        """Get the number of parameters in the saving function.

        Defaults to 1 parameter per variable in the data. This method should be
        overwritten in subclasses if the cost function has a different number of
        parameters per variable.

        Parameters
        ----------
        p : int
            Number of variables in the data.
        """
        return p

    def _fit(self, X: ArrayLike, y=None):
        """Fit the saving evaluator.

        Parameters
        ----------
        X : array-like
            Input data.
        y : None
            Ignored. Included for API consistency by convention.

        Returns
        -------
        self :
            Reference to self.
        """
        self.sums_ = col_cumsum(X, init_zero=True)
        return self

    def _evaluate(self, cuts: np.ndarray) -> np.ndarray:
        """Evaluate the saving on a set of intervals.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array with two columns of integer location-based intervals to evaluate.
            The subsets X[cuts[i, 0]:cuts[i, 1]] for
            i = 0, ..., len(cuts) are evaluated.

        Returns
        -------
        savings : np.ndarray
            A 2D array of savings. One row for each interval. The number of
            columns is 1 if the saving is inherently multivariate. The number of
            columns is equal to the number of columns in the input data if the saving is
            univariate. In this case, each column represents the univariate saving for
            the corresponding input data column.
        """
        starts = cuts[:, 0]
        ends = cuts[:, 1]
        return l2_saving(starts, ends, self.sums_)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for interval scorer.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [{}, {}]
        return params
