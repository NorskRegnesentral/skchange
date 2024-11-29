"""The CUSUM test statistic for a change in the mean."""

__author__ = ["johannvk"]

import numpy as np
from numpy.typing import ArrayLike

from skchange.change_scores.base import BaseChangeScore
from skchange.costs.gaussian_cov_cost import GaussianCovCost
from skchange.utils.numba import njit


@njit
def half_integer_digamma(twice_n: int) -> float:
    """Calculate the digamma function for half integer values, i.e. `twice_n/2`.

    The digamma function is the logarithmic derivative of the gamma function.
    This function is capable of calculating the
    digamma function for half integer values.

    Source: https://en.wikipedia.org/wiki/Digamma_function

    Parameters
    ----------
    twice_n : int
        Twice the integer value.

    Returns
    -------
    res : float
        Value of the digamma function for the half integer value.
    """
    assert twice_n > 0, "n must be a positive integer."

    if twice_n % 2 == 0:
        # Even integer: twice_n = 2n
        res = -np.euler_gamma
        n = twice_n // 2
        for k in range(0, n - 1):
            res += 1.0 / (k + 1.0)
    else:
        res = -2 * np.log(2) - np.euler_gamma
        # Odd integer: twice_n = 2n + 1
        n = (twice_n - 1) // 2
        for k in range(1, n + 1):
            res += 2.0 / (2.0 * k - 1.0)

    return res


@njit
def likelihood_ratio_expected_value(
    sequence_length: int, cut_point: int, dimension: int
) -> float:
    """Calculate the expected value of twice the negative log likelihood ratio.

    We check that the cut point is within the sequence length, and that both `k` and `n`
    are large enough relative to the dimension `p`, to ensure that the expected
    value is finite.
    Should at least have `p+1` points on each side of a split, for `p` dimensional data.

    Parameters
    ----------
    sequence_length : int
        Length of the sequence.
    cut_point : int
        Cut point of the sequence.
    dimension : int
        Dimension of the data.

    Returns
    -------
    g_k_n : float
        Expected value of twice the negative log likelihood ratio.
    """
    n, k, p = sequence_length, cut_point, dimension

    assert 0 < k < n, "Cut point `k` must be within the sequence length `n`."
    assert p > 0, "Dimension `p` must be a positive integer."
    assert k > (p + 1), "Cut point `k` must be larger than the dimension + 1."
    assert n - k > (
        p + 1
    ), "Run length after cut point `n - k` must be larger than dimension + 1."

    g_k_n = p * (
        np.log(2)
        + (n - 1) * np.log(n - 1)
        - (n - k - 1) * np.log(n - k - 1)
        - (k - 1) * np.log(k - 1)
    )

    for j in range(1, p + 1):
        g_k_n += (
            (n - 1) * half_integer_digamma(n - j)
            - (k - 1) * half_integer_digamma(k - j)
            - (n - k - 1) * half_integer_digamma(n - k - j)
        )

    return g_k_n


@njit
def compute_bartlett_corrections(
    sequence_lengths: np.ndarray, cut_points: np.ndarray, dimension: int
):
    """Calculate the Bartlett correction for the twice negated log likelihood ratio.

    Parameters
    ----------
    twice_negated_log_lr : float
        Twice the negative log likelihood ratio.
    sequence_length : int
        Length of the sequence.
    cut_point : int
        Cut point of the sequence.
    dimension : int
        Dimension of the data.

    Returns
    -------
    bartlett_corr_log_lr : float
    """
    bartlett_corrections = np.zeros(
        shape=(sequence_lengths.shape[0], 1), dtype=np.float64
    )

    for i, (sequence_length, cut_point) in enumerate(zip(sequence_lengths, cut_points)):
        g_k_n = likelihood_ratio_expected_value(
            sequence_length=sequence_length, cut_point=cut_point, dimension=dimension
        )
        bartlett_correction_factor = dimension * (dimension + 3.0) / g_k_n
        bartlett_corrections[i] = bartlett_correction_factor

    return bartlett_corrections


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
        total_intervals = cuts[:, [0, 2]]

        raw_scores = self._gaussian_cov_cost.evaluate(total_intervals) - (
            self._gaussian_cov_cost.evaluate(start_intervals)
            + self._gaussian_cov_cost.evaluate(end_intervals)
        )

        if self._apply_bartlett_correction:
            segment_lengths = cuts[:, 2] - cuts[:, 0]
            segment_splits = cuts[:, 1] - cuts[:, 0]
            bartlett_corrections = compute_bartlett_corrections(
                sequence_lengths=segment_lengths,
                cut_points=segment_splits,
                dimension=self._gaussian_cov_cost.dimension,
            )
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
