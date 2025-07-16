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
- you can add more private methods, but do not override BaseIntervalScorer's private
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
    number of parameters      - get_model_size(self, p)

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()

copyright: skchange developers, BSD-3-Clause License (see LICENSE file)
"""

# todo for internal extensions: copy this file to skchange/change_scores and rename it
# to _<your_score_name>.py

# todo: add any necessary imports
import numpy as np

# internal extensions in skchange.change_scores
from ..base import BaseIntervalScorer

# For external extensions, use the following imports:
# from skchange.base import BaseIntervalScorer

# todo: add the score to the CHANGE_SCORES variable in
# skchange.change_scores.__init__.py


class MyChangeScore(BaseIntervalScorer):
    """Custom score class.

    todo: write docstring, describing your custom score.

    Parameters
    ----------
    param1 : string, optional, default=None
        descriptive explanation of param1
    param2 : float, optional, default=1.0
        descriptive explanation of param2
    and so on
    """

    # todo: set tags
    _tags = {
        "authors": ["Tveten", "johannvk"],  # Use Github username
        "maintainers": "Tveten",  # Use Github username
        "task": "change_score",
        # distribution_type is used to automatically create test cases.
        # Valid values:
        # - "None" - No distributional restrictions. Test data: Mostly Gaussian, but
        #   no guarantee.
        # - "Poisson" - Integer data. Test data: Poisson distributed.
        # - "Gaussian" - Real-valued data. Test data: Gaussian distribution.
        "distribution_type": "None",  # "None", "Poisson", "Gaussian"
        # is_conditional: whether the scorer uses some of the input variables as
        # covariates in a regression model or similar. If `True`, the scorer requires
        # at least two input variables. If `False`, all p input variables/columns are
        # used to evaluate the score, such that the output has either 1 or p columns.
        "is_conditional": False,
        # is_aggregated: whether the scorer always returns a single value per cut or
        # not, irrespective of the input data shape.
        # Many scorers will not be aggregated, for example all scorers that evaluate
        # each input variable separately and return a score vector with one score for
        # each variable.
        "is_aggregated": False,
        # is_penalised: indicates whether the score is inherently penalised (True) or
        # not (False). If `True`, a score > 0 means that a change or anomaly is
        # detected. Penalised scores can be both positive and negative.
        # If `False`, the score is not penalised. To test for the existence of a change,
        # penalisation must be performed externally. Such scores are always
        # non-negative.
        "is_penalised": False,
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(
        self,
        param1=None,  # Custom parameter 1.
        param2=1.0,  # Custom parameter 2.
    ):
        # todo: write any hyper-parameters and components to self. These should never
        # be overwritten in other methods.
        # estimators should precede parameters
        #  if estimators have default values, set None and initialize below
        self.param1 = param1
        self.param2 = param2

        # leave this as is.
        super().__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._param1, etc.
        if self.param2 is None:
            from skchange.somewhere import MyOtherScorer

            self._param2 = MyOtherScorer(foo=42)
        else:
            # estimators should be cloned to avoid side effects
            self._param2 = param2.clone()

        # todo: if tags of estimator depend on component tags, set these here
        #  only needed if estimator is a composite
        #  tags set in the constructor apply to the object and override the class
        #
        # example 1: conditional setting of a tag
        # if est.foo == 42:
        #   self.set_tags(handles-missing-data=True)
        # example 2: cloning tags from component
        #   self.clone_tags(est2, ["enforce_index_type", "handles-missing-data"])

    # todo: implement, mandatory
    def _fit(self, X: np.ndarray, y=None):
        """Fit the score.

        This method precomputes quantities that speed up the score evaluation.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y: None
            Ignored. Included for API consistency by convention.
        """
        # todo: implement precomputations here
        # IMPORTANT: avoid side effects to X, y.

        # for example for CUSUM:
        # self._sums = col_cumsum(X, init_zero=True)

        return self

    # todo: implement, mandatory
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
        # todo: implement evaluation logic here. Must have the output format as
        #       described in the docstring.
        # IMPORTANT: avoid side effects to starts, ends.

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
        # For example for a mean and variance score:
        # return 2
        #
        # For example for a covariance matrix score:
        # if self.is_fitted:
        #     return self.n_variables + 1
        # else:
        #     return None

    # todo: implement, optional, defaults to output p (one parameter per variable).
    # used for setting a decent default penalty in detectors.
    def get_model_size(self, p: int) -> int:
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
        # For example for a covariance matrix score:
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
            There are currently no reserved values for change scores.

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
