"""The CUSUM test statistic for a change in the mean."""

__author__ = ["johannvk"]

import numpy as np
from numpy.typing import ArrayLike

from skchange.change_scores.base import BaseChangeScore
from skchange.costs.gaussian_cov_cost import GaussianCovCost

# TODO: Retrieve Bartlett correction functions.


def bartlett_correction(
    start_intervals: np.ndarray,
    end_intervals: np.ndarray,
    num_vars: int,
) -> np.ndarray:
    """Calculate the Bartlett correction for the Gaussian covariance change score."""
    pass


class GaussianCovScore(BaseChangeScore):
    """Gaussian covariance change score for a change in mean and/or covariance."""

    def __init__(
        self, cache_covariance: bool = False, apply_bartlett_correction: bool = True
    ):
        super().__init__()
        self._gaussian_cov_cost = GaussianCovCost()
        self._cache_covariance = cache_covariance
        self._apply_bartlett_correction = apply_bartlett_correction

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate."""
        if self._is_fitted:
            return self._gaussian_cov_cost.min_size
        else:
            return None

    def _fit(self, X: ArrayLike, y=None):
        """Fit the change score evaluator.

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
        self._gaussian_cov_cost.fit(X)
        return self

    def _evaluate(self, cuts: np.ndarray):
        """Evaluate the change score for a split within an interval.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array with three columns of integer locations.
            The first column is the start, the second is the split, and the third is
            the end of the interval to evaluate.
            The difference between subsets X[start:split] and X[split:end] is evaluated
            for each row in cuts.

        Returns
        -------
        scores : np.ndarray
            A 2D array of change scores. One row for each cut. The number of
            columns is 1 if the change score is inherently multivariate. The number of
            columns is equal to the number of columns in the input data if the score is
            univariate. In this case, each column represents the univariate score for
            the corresponding input data column.
        """
        start_intervals = cuts[:, [0, 1]]
        end_intervals = cuts[:, [1, 2]]
        raw_scores = self._gaussian_cov_cost.evaluate(
            start_intervals
        ) - self._gaussian_cov_cost.evaluate(end_intervals)

        if self._apply_bartlett_correction:
            raise NotImplementedError("Bartlett correction not yet implemented.")
            bartlett_corrections = 2
            return bartlett_corrections * raw_scores
        else:
            return raw_scores

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for interval scorers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        raise NotImplementedError("Test parameters not yet implemented.")
