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

from skchange.scores.mean_score import init_mean_score, mean_score

VALID_SCORES = ["mean"]


def score_factory(score_name: str):
    """Return score function and its initializer.

    Parameters
    ----------
    score_name : str
        Name of score function. Must be one of 'mean'.

    Returns
    -------
    score_func : Callable
        Score function.
    init_score_func : Callable
        Score function initializer.
    """
    if score_name == "mean":
        return mean_score, init_mean_score
    else:
        message = (
            f"score_name={score_name} not recognized."
            + f" Must be one of {', '.join(VALID_SCORES)}"
        )
        raise ValueError(message)
