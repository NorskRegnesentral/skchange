"""Factory for getting test statistic functions and their initializers from strings.

Recipe for adding new scores (replace "score" with "score" below):
    1. Add a new module in skchange.score: <score_name>_score.py
    2. Add two functions:
        init_<score_name>_score, which precomputes quantities that should be
        precomputed, often partial sums and such.
        <score_name>_score, which takes in the output of init_<score_name>_score as its
        first argument, and start, end and split indices as the second, third and
        fourth arguments.
    3. Add the name of the score to VALID_SCORES below.
    4. Add the name of the score to the docstring of score_factory below.
    4. Add a new if-statement in score_factory below.

"""

from skchange.scores.mean_score import init_mean_score, mean_score
from skchange.scores.var_score import init_var_score, var_score

VALID_SCORES = ["mean", "var"]


def score_factory(score_name: str):
    """Return score function and its initializer.

    Parameters
    ----------
    score_name : str
        Name of score function. Must be one of 'mean' or 'var'.

    Returns
    -------
    score_func : Callable
        Score function.
    init_score_func : Callable
        Score function initializer.
    """
    if score_name == "mean":
        return mean_score, init_mean_score
    elif score_name == "var":
        return var_score, init_var_score
    else:
        message = (
            f"score_name={score_name} not recognized."
            + f" Must be one of {', '.join(VALID_SCORES)}"
        )
        raise ValueError(message)
