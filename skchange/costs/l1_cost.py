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
from skchange.costs.utils import check_mean, check_univariate_scale
from skchange.utils.numba import njit
from skchange.utils.numba.stats import col_median
from skchange.utils.validation.enums import EvaluationType


@njit
def laplace_log_likelihood(centered_X: np.ndarray, scale: float) -> float:
    """Evaluate the log likelihood of a Laplace distribution.

    Parameters
    ----------
    centered_X : np.ndarray
        Column of data to evaluate the log likelihood for.
    scale : float
        Scale parameter of the Laplace distribution.

    Returns
    -------
    log_likelihood : float
        Log likelihood of centered_X sampled i.i.d from a Laplace distribution
        with location = 0.0 and scale parameter as specified.
    """
    n_samples = len(centered_X)
    return -n_samples * np.log(2 * scale) - np.sum(np.abs(centered_X)) / scale


@njit
def laplace_cost_mle_params(starts: np.ndarray, ends: np.ndarray, X: np.ndarray):
    """Evaluate the L1 cost for the optimal parameters.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data, where each column
        represents the univariate cost for the corresponding input data column.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        mle_locations = col_median(X[start:end])
        centered_X = X[start:end, :] - mle_locations[None, :]

        for j in range(n_columns):
            mle_scale = np.mean(np.abs(centered_X[:, j]))
            costs[i, j] = -2.0 * laplace_log_likelihood(centered_X[:, j], mle_scale)

    return costs


def laplace_cost_mle_location_fixed_scale(starts, ends, X, scales: np.ndarray):
    """Evaluate the L1 cost for a known scale.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data, where each column
        represents the univariate cost for the corresponding input data column.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        mle_locations = col_median(X[start:end])
        centered_X = X[start:end, :] - mle_locations[None, :]
        for j in range(n_columns):
            costs[i, j] = -2.0 * laplace_log_likelihood(centered_X[:, j], scales[j])

    return costs


@njit
def laplace_cost_fixed_params(
    starts, ends, X, locations: np.ndarray, scales: np.ndarray
):
    """Evaluate the L1 cost for fixed parameters.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data, where each column
        represents the univariate cost for the corresponding input data column.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        centered_X = X[start:end, :] - locations[None, :]
        for j in range(n_columns):
            costs[i, j] = -2.0 * laplace_log_likelihood(centered_X[:, j], scales[j])

    return costs


class L1Cost(BaseCost):
    """L1 cost for a constant Laplace distribution.

    Parameters
    ----------
    param : any, optional (default=None)
        If None, the cost is evaluated for an interval-optimised parameter, often the
        maximum likelihood estimate. If not None, the cost is evaluated for the
        specified fixed parameter.
    known_scale : bool, optional (default=False)
        Whether the scale parameter of the Laplace distribution is known or not.
        If it is known, the cost is evaluated for the fixed scale parameter(s).
    """

    evaluation_type = EvaluationType.UNIVARIATE
    supports_fixed_params = True

    def __init__(
        self,
        param=None,  # Mandatory first parameter (see docs above).
        known_scale: bool = False,  # Optional parameter with default value.
    ):
        super().__init__(param)
        self.known_scale = known_scale
        self._locations = None
        self._scales = None
        # param: (location, scale) of the Laplace distribution.
        # (mean & median, diversity).

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
        if self.param is not None:
            if self.known_scale:
                self._scales = self._param
            else:
                self._locations, self._scales = self._param

        # MLE of the Laplace distribution location parameter is the median.
        # Tricky to precompute the median, so we'll do it on the fly.

        return self

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
            columns is equal to the number of columns in the input data
            passed to `.fit()`. Each column represents the univariate
            cost for the corresponding input data column.
        """
        return laplace_cost_mle_params(starts, ends, self._X)

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
            columns is equal to the number of columns in the input data
            passed to `.fit()`. Each column represents the univariate
            cost for the corresponding input data column.
        """
        if self.known_scale:
            return laplace_cost_mle_location_fixed_scale(
                starts, ends, self._X, self._scales
            )
        else:
            return laplace_cost_fixed_params(
                starts, ends, self._X, self._locations, self._scales
            )

    def _check_fixed_param(self, param: tuple, X: np.ndarray) -> np.ndarray:
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
        if self.known_scale:
            if isinstance(param, tuple):
                if len(param) != 1:
                    raise ValueError(
                        "Fixed parameter must be (scale) when 'known_scale == True'."
                    )
                else:
                    return check_univariate_scale(param[0], X)
            elif not isinstance(param, tuple):
                return check_univariate_scale(param, X)
        else:
            if not isinstance(param, tuple) or len(param) != 2:
                raise ValueError(
                    "Fixed parameter must be (location, scale) when"
                    " 'known_scale == False'."
                )
            means = check_mean(param[0], X)
            scales = check_univariate_scale(param[1], X)
            return means, scales

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
        return p if self.known_scale else 2 * p

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
