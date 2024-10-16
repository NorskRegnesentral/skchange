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

from typing import Callable, Union

from numba.extending import is_jitted

from skchange.costs.mean_cost import init_mean_cost, mean_cost

VALID_COSTS = ["mean"]


def cost_factory(cost: Union[str, tuple[Callable, Callable]]):
    """Return cost function and its initializer.

    Parameters
    ----------
    cost : {"mean"}, tuple[Callable, Callable], default="mean
        Name of cost function to use for changepoint detection.

        * `"mean"`: The Gaussian mean likelihood cost is used.
        * If a tuple, it must contain two numba jitted functions:

            1. The first function is the cost function, which takes three arguments:

                1. `precomputed_params`: The output of the second function.
                2. `starts`: Start indices of the intervals to calculate the cost for.
                3. `ends`: End indices of the intervals to calculate the cost for.

            The algorithms that use the cost function govern what intervals are
            considered.

            2. The second function is the initializer, which takes the data matrix as
               input and returns precomputed quantities that may speed up the cost
               calculations. If not relevant, just return the input data matrix.

    Returns
    -------
    cost_func : Callable
        Cost function.
    init_cost_func : Callable
        Cost function initializer.

    Raises
    ------
    ValueError
        If the provided `cost` is not recognized, an error is raised with a message
        indicating the valid options.

    """
    if cost == "mean":
        return mean_cost, init_mean_cost
    elif len(cost) == 2 and all([is_jitted(s) for s in cost]):
        return cost[0], cost[1]
    else:
        message = (
            f"`cost`={cost} not recognized."
            + f" Must be one of {', '.join(VALID_COSTS)}"
            + " or a tuple of two numba jitted functions."
        )
        raise ValueError(message)
