"""Factory for getting test statistic functions and their initializers from strings.

Recipe for adding new test statistics (replace "cost" with "test statistic" below):
    1. Add a new module in skchange.costs: <cost_name>_cost.py
    2. Add two functions:
        init_<cost_name>_cost, which precomputes quantities that should be precomputed,
        often partial sums and such.
        <cost_name>_cost, which takes in the output of init_<cost_name>_cost as its
        first argument, and starts and ends indices of the costs to be computed as its
        second and third arguments.
    3. Add the name of the cost to VALID_COSTS below.
    4. Add the name of the cost to the docstring of cost_factory below.
    4. Add a new if-statement in cost_factory below.

"""

from skchange.test_stats.mean_test_stat import init_mean_test_stat, mean_test_stat

VALID_TEST_STATS = ["mean"]


def test_stat_factory(test_stat_name: str):
    if test_stat_name == "mean":
        return mean_test_stat, init_mean_test_stat
    else:
        message = (
            f"test_stat_name={test_stat_name} not recognized."
            + f" Must be one of {', '.join(VALID_TEST_STATS)}"
        )
        raise ValueError(message)
