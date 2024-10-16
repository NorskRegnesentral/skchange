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

__author__ = ["Tveten"]

from typing import Callable, Union

from numba.extending import is_jitted

from skchange.costs.mean_saving import init_mean_saving, mean_saving

VALID_SAVINGS = ["mean"]


def saving_factory(saving: Union[str, tuple[Callable, Callable]]):
    """Return saving function and its initializer.

    Parameters
    ----------
    saving : {"mean"} or `tuple[Callable, Callable]`, default="mean"
        Name of saving function to use for anomaly detection.

        * `"mean"`: The Gaussian mean likelihood cost is used.
        * If a tuple, it must contain two numba jitted functions:

            1. The first function is the saving function, which takes three arguments:

                1. `precomputed_params`: The output of the second function.
                2. `starts`: Start indices of the intervals to calculate the saving for.
                3. `ends`: End indices of the intervals to calculate the saving for.

            The algorithms that use the saving function govern what intervals are
            considered.

            2. The second function is the initializer, which takes the data matrix as
               input and returns precomputed quantities that may speed up the saving
               calculations. If not relevant, just return the input data matrix.

    Returns
    -------
    saving_func : `Callable`
        Saving function.
    init_saving_func : `Callable`
        Saving function initializer.

    Raises
    ------
    ValueError
        If `saving` is not recognized, an error is raised with a message indicating the
        valid options.
    """
    if saving == "mean":
        return mean_saving, init_mean_saving
    elif len(saving) == 2 and all([is_jitted(s) for s in saving]):
        return saving[0], saving[1]
    else:
        message = (
            f"saving={saving} not recognized."
            + f" Must be one of {', '.join(VALID_SAVINGS)}"
            + " or a tuple of two numba jitted functions."
        )
        raise ValueError(message)
