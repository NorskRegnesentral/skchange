"""Extension template for change scores as interval scorers.

Adapted from the sktime extension templates.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- you can add more private methods, but do not override BaseChangeScore's private
    methods. An easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by sktime.utils.estimator_checks.check_estimator
- once complete: use as a local library, or contribute to skchange via PR
- more details:
  https://www.sktime.net/en/stable/developer_guide/add_estimators.html

Mandatory implements:
    fitting                  - _fit(self, X, y=None)
    evaluating optimal param - _evaluate(self, cuts)

Optional implements:
    minimum size of interval  - min_size(self)
    number of parameters      - get_param_size(self, p)

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()

copyright: skchange developers, BSD-3-Clause License (see LICENSE file)
"""

import numpy as np
from scipy.stats import norm

from ..base import BaseIntervalScorer
from ..utils.numba import njit
from ..utils.numba.stats import col_cumsum
from ._cusum import cusum_score


# TODO: document this
@njit
def ESAC_Threshold(
    cusum_scores: np.ndarray,
    a_s: np.ndarray,
    nu_s: np.ndarray,
    t_s: np.ndarray,
    threshold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compute ESAC scores from CUSUM scores.

    This function calculates the penalised score for the ESAC algorithm,
    as defined in Equation (6) in [1]_. The outputs are penalised CUSUM
    scores computed from the input CUSUM scores.

    Parameters
    ----------
        cusum_scores (np.ndarray): A 2D array where each row represents the
            CUSUM scores for a specific time step.
        a_s (np.ndarray): A 1D array of hard threshold values. Correspond to
            a(t) as defined in Equation (4) in [1]_ for each t specified in t_s.
        nu_s (np.ndarray): A 1D array of mean-centering terms. Correspond to
            nu(t) as defined after Equation (4) in [1]_ for each t specified in t_s.
        t_s (np.ndarray): A 1D array of candidate sparsity values corresponding
            to the element in a_s and nu_s.
        threshold (np.ndarray): A 1D array of penalty values, corresponding to \gamma(t)
            in Equation (4) in [1]_, where t is as defined in t_s.

    Returns
    -------
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - output_scores (np.ndarray): A 2D array of computed ESAC scores. The array
                has one column.
            - sargmax (np.ndarray): A 1D array of indices or labels corresponding
                to the sparisty level at which the maximum score was achieved.

    References
    ----------
    ..  [1] Per August Jarval Moen, Ingrid Kristine Glad, Martin Tveten. Efficient
        sparsity adaptive changepoint estimation. Electron. J. Statist. 18 (2)
        3975 - 4038, 2024. https://doi.org/10.1214/24-EJS2294.
    """
    num_levels = len(threshold)
    num_cusum_scores = len(cusum_scores)
    output_scores = np.zeros(num_cusum_scores, dtype=np.float64)
    sargmax = np.zeros(num_cusum_scores, dtype=np.int64)

    for i in range(num_cusum_scores):
        temp_max = -np.inf
        for j in range(num_levels):
            temp_vec = (cusum_scores[i])[np.abs(cusum_scores[i]) > a_s[j]]
            if len(temp_vec) > 0:
                temp = np.sum(temp_vec**2 - nu_s[j]) - threshold[j]
                if temp > temp_max:
                    temp_max = temp
                    sargmax[i] = t_s[j]

        output_scores[i] = temp_max

    return output_scores.reshape(-1, 1), sargmax


class ESACScore(BaseIntervalScorer):
    """Custom score class.

    This is the sparsity adaptive penalised CUSUM score for a change in the mean.
    The ESAC score is a penalised version of the CUSUM score, where the CUSUM of
    each time series is thresholded, mean-centered and penalised by a sparsity
    dependant penalty. The score is defined in Equation (6) in [1]_.

    Parameters
    ----------
    threshold_dense : float, optional, default=1.5
        The leading constant in the penalty function taken as in (8) in [1]_ in the
        dense case where the candidate sparisity level t is greater than or equal to
        sqrt(p * log(n)).
    threshold_sparse : float, optional, default=1.0
        The leading constant in the penalty function taken as in (8) in [1]_ in the
        sparse case where the candidate sparisity level t is less than sqrt(p * log(n)).

    References
    ----------
    ..  [1] Per August Jarval Moen, Ingrid Kristine Glad, Martin Tveten. Efficient
        sparsity adaptive changepoint estimation. Electron. J. Statist. 18 (2)
        3975 - 4038, 2024. https://doi.org/10.1214/24-EJS2294.
    """

    _tags = {
        "authors": ["peraugustmoen", "Tveten"],
        "maintainers": ["peraugustmoen", "Tveten"],
        "task": "change_score",
        "distribution_type": "Gaussian",
        "is_conditional": False,
        "is_aggregated": True,
        "is_penalised": True,
    }

    def __init__(
        self,
        threshold_dense=1.5,
        threshold_sparse=1.0,
    ):
        super().__init__()
        self.threshold_dense = threshold_dense
        self.threshold_sparse = threshold_sparse

    def _fit(self, X: np.ndarray, y=None):
        """Fit the change score evaluator.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y : None
            Ignored. Included for API consistency by convention.

        Returns
        -------
        self :
            Reference to self.
        """
        self._sums = col_cumsum(X, init_zero=True)
        self.n = X.shape[0]
        self.p = X.shape[1]
        n = self.n
        p = self.p

        if p == 1:
            self.a_s = np.array([0.0])
            self.nu_s = np.array([1.0])
            self.t_s = np.array([1])
            self.threshold = np.array(
                [self.threshold_dense * (np.sqrt(p * np.log(n)) + np.log(n))]
            )
        else:
            max_s = min(np.sqrt(p * np.log(n)), p)

            # Generate log2ss from 0 to floor(log2(max_s))
            log2ss = np.arange(0, np.floor(np.log2(max_s)) + 1)

            # calculate candidate sparsity values
            ss = 2**log2ss

            # add p as candidate sparsity and reverse order
            ss = np.concatenate(([p], ss[::-1]))
            self.t_s = np.array(ss, dtype=float)
            ss = np.array(ss, dtype=float)

            # Initialize as array
            self.a_s = np.zeros_like(ss, dtype=float)
            self.a_s[1:] = np.sqrt(
                2 * np.log(np.exp(1) * p * 4 * np.log(n) / ss[1:] ** 2)
            )

            # Calculate nu_as
            # nu_as = 1 + as * exp(dnorm(as, log=TRUE) - pnorm(as, lower.tail=FALSE,
            #   log.p=TRUE))
            log_dnorm = norm.logpdf(self.a_s)

            # Log of upper tail probability: log(1 - \Phi(as))
            log_pnorm_upper = norm.logsf(self.a_s)

            self.nu_s = 1 + self.a_s * np.exp(log_dnorm - log_pnorm_upper)

            self.threshold = np.zeros_like(ss, dtype=float)
            self.threshold[0] = self.threshold_dense * (
                np.sqrt(4 * p * np.log(n)) + 4 * np.log(n)
            )
            self.threshold[1:] = self.threshold_sparse * (
                ss[1:] * np.log(np.exp(1) * p * 4 * np.log(n) / ss[1:] ** 2)
                + 4 * np.log(n)
            )

        return self

    def _evaluate(self, cuts: np.ndarray):
        """Evaluate the change score for a split within an interval.

        Evaluates the score for `X[start:split]` vs. `X[split:end]` for each
        start, split, end in cuts.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array with three columns of integer locations.
            The first column is the ``start``, the second is the ``split``, and the
            third is the ``end`` of the interval to evaluate.
            The difference between subsets ``X[start:split]`` and ``X[split:end]`` is
            evaluated for each row in `cuts`.

        Returns
        -------
        scores : np.ndarray
            A 2D array of change scores. One row for each cut. The number of
            columns is 1 if the change score is inherently multivariate. The number of
            columns is equal to the number of columns in the input data if the score is
            univariate. In this case, each column represents the univariate score for
            the corresponding input data column.
        """
        starts = cuts[:, 0]
        splits = cuts[:, 1]
        ends = cuts[:, 2]
        cusum_scores = cusum_score(starts, ends, splits, self._sums)
        thresholded_cusum_scores, sargmaxes = ESAC_Threshold(
            cusum_scores, self.a_s, self.nu_s, self.t_s, self.threshold
        )
        self.sargmaxes = sargmaxes
        return thresholded_cusum_scores

    @property
    def min_size(self) -> int | None:
        """Minimum size of the interval to evaluate.

        The size of each interval is defined as ``cuts[i, 1] - cuts[i, 0]``.

        Returns
        -------
        int or None
            The minimum valid size of an interval to evaluate. If ``None``, it is
            unknown what the minimum size is. E.g., the scorer may need to be fitted
            first to determine the minimum size.
        """
        return 1

    def get_param_size(self, p: int) -> int:
        """Get the number of parameters estimated by the score in each segment.

        Parameters
        ----------
        p : int
            Number of variables in the data.

        Returns
        -------
        int
            Number of parameters in the score function.
        """
        return p

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for change scores.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            {"threshold_dense": 1.5, "threshold_sparse": 1.0},
            {"threshold_dense": 2.0, "threshold_sparse": 2.0},
        ]
        return params
