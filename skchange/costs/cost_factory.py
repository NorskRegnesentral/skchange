"""Factory for getting cost functions and their initializers from strings.

Recipe for adding new costs:
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

__author__ = ["Tveten"]

from skchange.costs.mean_cost import init_mean_cost, mean_cost

VALID_COSTS = ["mean"]


def cost_factory(cost_name: str):
    """Return cost function and its initializer.

    Parameters
    ----------
    cost_name : str
        Name of cost function. Must be one of 'mean'.

    Returns
    -------
    cost_func : Callable
        Cost function.
    init_cost_func : Callable
        Cost function initializer.
    """
    if cost_name == "mean":
        return mean_cost, init_mean_cost
    else:
        message = (
            f"cost_name={cost_name} not recognized."
            + f" Must be one of {', '.join(VALID_COSTS)}"
        )
        raise ValueError(message)
