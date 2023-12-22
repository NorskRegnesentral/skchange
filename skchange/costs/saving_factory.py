"""Factory for getting saving functions and their initializers from strings.

Savings are cost differences between a baseline cost and an optimised alternative cost.

Recipe for adding new savings:
    1. Add a new module in skchange.costs: <saving_name>_saving.py
    2. Add two functions:
        init_<saving_name>_saving, which precomputes quantities that should be
        precomputed, often partial sums and such.
        <saving_name>_saving, which takes in the output of init_<saving_name>_saving as
        its first argument, and starts and ends indices of the savings to be computed as
        its second and third arguments.
    3. Add the name of the saving to VALID_SAVINGS below.
    4. Add the name of the saving to the docstring of saving_factory below.
    4. Add a new if-statement in saving_factory below.

"""

from skchange.costs.mean_saving import init_mean_saving, mean_saving

VALID_SAVINGS = ["mean"]


def saving_factory(saving_name: str):
    """Return saving function and its initializer.

    Parameters
    ----------
    saving_name : str
        Name of saving function. Must be one of 'mean'.

    Returns
    -------
    cost_func : Callable
        Cost function.
    init_cost_func : Callable
        Cost function initializer.
    """
    if saving_name == "mean":
        return mean_saving, init_mean_saving
    else:
        message = (
            f"saving_name={saving_name} not recognized."
            + f" Must be one of {', '.join(VALID_SAVINGS)}"
        )
        raise ValueError(message)
