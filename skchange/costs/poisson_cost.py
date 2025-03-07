"""Extension template for costs as interval scorers.

Adapted from the sktime extension templates.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- you can add more private methods, but do not override BaseCost's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by sktime.utils.estimator_checks.check_estimator
- once complete: use as a local library, or contribute to skchange via PR
- more details:
  https://www.sktime.net/en/stable/developer_guide/add_estimators.html

Mandatory implements:
    fitting                  - _fit(self, X, y=None)
    evaluating optimal param - _evaluate_optim_param(self, starts, ends)

Optional implements:
    evaluating fixed param   - _evaluate_fixed_param(self, starts, ends)
    checking fixed param     - _check_fixed_param(self, param, X)
    minimum size of interval  - min_size(self)
    number of parameters      - get_param_size(self, p)

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()

copyright: skchange developers, BSD-3-Clause License (see LICENSE file)
"""

import numpy as np

from skchange.costs import BaseCost
from skchange.utils.validation.enums import EvaluationType

# todo: add any necessary imports here

# todo: add the cost to the COSTS variable in skchange/costs/__init__.py

import scipy.stats as stats

stats.poisson.ppf(0.01, 1)

poisson_sample = stats.poisson.rvs(5, size=1000)
mle_rate = np.mean(poisson_sample)

# %%
from skchange.utils.numba import njit, prange


@njit
def poisson_log_likelihood(rate, samples):
    # Assume sample: np.ndarray[int]
    max_sample = np.max(samples)
    if max_sample < 2:
        sum_log_factorial_samples = 0
    else:
        sample_hist = np.zeros(max_sample - 1, dtype="int")

        for sample_idx in prange(len(samples)):
            sample = samples[sample_idx]
            # Increment the count of the sample values
            # (ignoring the values 0, and 1, as log(0!) = log(1!) = 0.)
            sample_hist[: (sample - 1)] += 1
        sum_log_factorial_samples = np.sum(
            np.log(np.arange(2, max_sample + 1)) * sample_hist
        )

    return (
        -len(samples) * rate
        + np.log(rate) * np.sum(samples)
        - sum_log_factorial_samples
    )


@njit
def fast_poisson_log_likelihood(rate, samples):
    # Assume sample: np.ndarray[int]
    bin_counts = np.bincount(samples)
    sample_counts = np.flip(np.cumsum(np.flip(bin_counts)))
    max_sample_plus_one = len(sample_counts)

    if max_sample_plus_one < 2:
        sum_log_factorial_samples = 0
    else:
        sum_log_factorial_samples = np.sum(
            np.log(np.arange(2, max_sample_plus_one)) * sample_counts[2:]
        )

    return (
        -len(samples) * rate
        + np.log(rate) * np.sum(samples)
        - sum_log_factorial_samples
    )

@njit
def poisson_mle_rate_log_likelihood(mle_rate, samples):
    # Assume sample: np.ndarray[int]
    max_sample = np.max(samples)
    if max_sample < 2:
        sum_log_factorial_samples = 0
    else:
        sample_hist = np.zeros(max_sample - 1, dtype="int")

        for sample_index in prange(len(samples)):
            sample = samples[sample_index]
            # Increment the count of the sample values
            # (ignoring the values 0, and 1, as log(0!) = log(1!) = 0.)
            sample_hist[: (sample - 1)] += 1
        sum_log_factorial_samples = np.sum(
            np.log(np.arange(2, max_sample + 1)) * sample_hist
        )
    return (
        len(samples) * mle_rate * (np.log(mle_rate) - 1.0) - sum_log_factorial_samples
    )

@njit
def fast_poisson_mle_rate_log_likelihood(mle_rate, samples):
    # Assume sample: np.ndarray[int]
    bin_counts = np.bincount(samples)
    sample_counts = np.flip(np.cumsum(np.flip(bin_counts)))
    max_sample_plus_one = len(sample_counts)

    if max_sample_plus_one < 2:
        sum_log_factorial_samples = 0
    else:
        sum_log_factorial_samples = np.sum(
            np.log(np.arange(2, max_sample_plus_one)) * sample_counts[2:]
        )

    return (
        len(samples) * mle_rate * (np.log(mle_rate) - 1.0) - sum_log_factorial_samples
    )

@njit
def non_prange_poisson_mle_rate_log_likelihood(mle_rate, samples):
    # Assume sample: np.ndarray[int]
    max_sample = np.max(samples)
    if max_sample < 2:
        sum_log_factorial_samples = 0
    else:
        value_counts = np.zeros(max_sample - 1, dtype="int")

        for sample in samples:
            # Increment the count of the sample values
            # (ignoring the values 0, and 1, as log(0!) = log(1!) = 0.)
            value_counts[sample - 1] += 1

        sum_log_factorial_samples = np.sum(
            np.log(np.arange(2, max_sample + 1)) * value_counts
        )
    return (
        len(samples) * mle_rate * (np.log(mle_rate) - 1.0) - sum_log_factorial_samples
    )


def construct_sample_hist_2_and_above(samples):
    max_sample = np.max(samples)

    sample_hist = np.zeros(max_sample - 1, dtype="int")

    for sample in samples:
        # Increment the count of the sample values
        # (ignoring the values 0, and 1, as log(0!) = log(1!) = 0.)
        sample_hist[: (sample - 1)] += 1

    return sample_hist


def construct_sample_hist_2_and_above_v2(samples):
    max_sample = np.max(samples)

    sample_hist = np.zeros(max_sample, dtype="int")

    for sample in samples:
        # Increment the count of the sample values
        # (ignoring the values 0, and 1, as log(0!) = log(1!) = 0.)
        sample_hist[sample - 1] += 1

    # Cumsum from the end of the array:
    sample_counts = np.flip(np.cumsum(np.flip(sample_hist)))

    return sample_counts, sample_hist

def construct_sample_hist_2_and_above_v3(samples):
    # Cumsum from the end of the array:
    sample_counts = np.flip(np.cumsum(np.flip(np.bincount(samples)[2:])))

    return sample_counts

sample_hist_2_and_above = construct_sample_hist_2_and_above(poisson_sample)
sample_counts_v2, sample_hist_v2 = construct_sample_hist_2_and_above_v2(poisson_sample)
sample_counts_v3 = construct_sample_hist_2_and_above_v3(poisson_sample)
np.bincount(poisson_sample)
# %%
stats.poisson.logpmf(poisson_sample, mle_rate).sum()
poisson_mle_rate_log_likelihood(mle_rate, poisson_sample)
poisson_log_likelihood(mle_rate, poisson_sample)
fast_poisson_log_likelihood(mle_rate, poisson_sample)
fast_poisson_mle_rate_log_likelihood(mle_rate, poisson_sample)

# %%
%%timeit
log_likelihood_cost = -stats.poisson.logpmf(poisson_sample, mle_rate).sum()

# %%
%%timeit
manual_poisson_ll_cost = -poisson_log_likelihood(mle_rate, poisson_sample)

# %%
%%timeit
manual_poisson_mle_ll_cost = -poisson_mle_rate_log_likelihood(mle_rate, poisson_sample)

# %%
%%timeit
fast_poisson_log_likelihood(mle_rate, poisson_sample)

# %%
%%timeit
fast_poisson_mle_rate_log_likelihood(mle_rate, poisson_sample)

# %%
poisson_cost = -len(poisson_sample) * np.log(mle_rate) * mle_rate


class MyCost(BaseCost):
    """Custom cost class.

    todo: write docstring, describing your custom cost.

    Parameters
    ----------
    param : any, optional (default=None)
        If None, the cost is evaluated for an interval-optimised parameter, often the
        maximum likelihood estimate. If not None, the cost is evaluated for the
        specified fixed parameter.
    param1 : string, optional, default=None
        descriptive explanation of param1
    param2 : float, optional, default=1.0
        descriptive explanation of param2
    and so on
    """

    # todo: add authors and maintaners Github user name
    _tags = {
        "authors": ["Tveten", "johannvk"],
        "maintainers": "Tveten",
    }

    # Does the cost evaluate univariate or multivariate data?
    # If the evaluation_type is EvaluationType.UNIVARIATE:
    #   * the cost is vectorized over columns in `X` input to `fit`.
    #   * the output of `evaluate` is has the same number of columns as `X`.
    # If the evaluation_type is EvaluationType.MULTIVARIATE:
    #   * the cost is evaluated on each row of `X` input to `fit`.
    #   * the output of `evaluate` is always a single column, one value per `cut`.
    evaluation_type = EvaluationType.UNIVARIATE
    # Does the cost support fixed parameters? I.e., is `_evaluate_fixed_param`
    # implemented? Fixed parameter evaluation is required for the cost to be used as a
    # saving in certain anomaly detectors.
    supports_fixed_params = False

    # todo: add any hyper-parameters and components to constructor
    # todo: if fixed parameters are supported, add type hints to `param`
    def __init__(
        self,
        param=None,  # Mandatory first parameter (see docs above).
        param1=None,  # Custom parameter 1.
        param2=1.0,  # Custom parameter 2.
    ):
        super().__init__(param)

        # todo: write any hyper-parameters and components to self. These should never
        # be overwritten in other methods.
        self.param1 = param1
        self.param2 = param2

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._param1, etc.

    # todo: implement, mandatory
    def _fit(self, X: np.ndarray, y=None):
        """Fit the cost.

        This method precomputes quantities that speed up the cost evaluation.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y: None
            Ignored. Included for API consistency by convention.
        """
        self._param = self._check_param(self.param, X)

        # todo: implement precomputations here
        # IMPORTANT: avoid side effects to X, y.

        # for example for the L2 cost:
        # self._sums = col_cumsum(X, init_zero=True)
        # self._sums2 = col_cumsum(X**2, init_zero=True)

        return self

    # todo: implement, mandatory
    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the optimal parameters.

        Evaluates the cost for `X[start:end]` for each each start, end in starts, ends.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of
            columns is 1 if the cost has `evaluation_type = "multivariate"`.
            The number of columns is equal to the number of columns in the input data if
            the `evaluation_type = "univariate"`. In this case, each column represents
            the univariate cost for the corresponding input data column.
        """
        # todo: implement evaluation logic here. Must have the output format as
        #       described in the docstring.
        # IMPORTANT: avoid side effects to starts, ends.

    # todo: implement, optional. Mandatory if supports_fixed_params = True.
    def _evaluate_fixed_param(self, starts, ends) -> np.ndarray:
        """Evaluate the cost for the fixed parameters.

        Evaluates the cost for `X[start:end]` for each each start, end in starts, ends.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of
            columns is 1 if the cost has `evaluation_type = "multivariate"`.
            The number of columns is equal to the number of columns in the input data if
            the `evaluation_type = "univariate"`. In this case, each column represents
            the univariate cost for the corresponding input data column.
        """
        # todo: implement evaluation logic here. Must have the output format as
        #       described in the docstring.
        # IMPORTANT: avoid side effects to starts, ends.

    # todo: implement, optional, defaults to no checking.
    # used inside self._check_param in _fit.
    def _check_fixed_param(self, param, X: np.ndarray) -> np.ndarray:
        """Check if the fixed parameter is valid relative to the data.

        Parameters
        ----------
        param : any
            Fixed parameter for the cost calculation.
        X : np.ndarray
            Input data.

        Returns
        -------
        param: any
            Fixed parameter for the cost calculation.
        """
        return param

    # todo: implement, optional, defaults to min_size = 1.
    # used for automatic validation of cuts in `evaluate`.
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
        # For example for a mean and variance cost:
        # return 2
        #
        # For example for a covariance matrix cost:
        # if self.is_fitted:
        #     return self._X.shape[1] + 1
        # else:
        #     return None

    # todo: implement, optional, defaults to output p (one parameter per variable).
    # used for setting a decent default penalty in detectors.
    def get_param_size(self, p: int) -> int:
        """Get the number of parameters in the cost function.

        Parameters
        ----------
        p : int
            Number of variables in the data.

        Returns
        -------
        int
            Number of parameters in the cost function.
        """
        # For example for a covariance matrix cost:
        # return p * (p + 1) // 2

    # todo: return default parameters, so that a test instance can be created.
    #       required for automated unit and integration testing of estimator.
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for costs.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        # Testing parameter choice should cover internal cases well.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # The parameter_set argument is not used for automated, module level tests.
        #   It can be used in custom, estimator specific tests, for "special" settings.
        # A parameter dictionary must be returned *for all values* of parameter_set,
        #   i.e., "parameter_set not available" errors should never be raised.
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        # return params
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        # return params
        #
        # example 3: parameter set depending on param_set value
        #   note: only needed if a separate parameter set is needed in tests
        # if parameter_set == "special_param_set":
        #     params = {"est": value1, "parama": value2}
        #     return params
        #
        # # "default" params - always returned except for "special_param_set" value
        # params = {"est": value3, "parama": value4}
        # return params
