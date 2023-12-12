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

from numba.extending import is_jitted

from skchange.scores.mean_score import init_mean_score, mean_score
from skchange.scores.var_score import init_var_score, var_score

VALID_SCORES = ["mean", "var"]


def score_factory(score: str):
    """Return score function and its initializer.

    Parameters
    ----------
    score: str, Tuple[Callable, Callable], optional (default="mean")
        Test statistic to use for changepoint detection.
        * If "mean", the difference-in-mean statistic is used,
        * If "var", the difference-in-variance statistic is used,
        * If a tuple, it must contain two functions: The first function is the scoring
        function, which takes in the output of the second function as its first
        argument, and start, end and split indices as the second, third and fourth
        arguments. The second function is the initializer, which precomputes quantities
        that should be precomputed. See skchange/scores/score_factory.py for examples.

    Returns
    -------
    score_func : Numba jitted Callable
        Score function.
    init_score_func : Numba jitted Callable
        Score function initializer.
    """
    if isinstance(score, str) and score == "mean":
        return mean_score, init_mean_score
    elif isinstance(score, str) and score == "var":
        return var_score, init_var_score
    elif len(score) == 2 and all([is_jitted(s) for s in score]):
        return score[0], score[1]
    else:
        message = (
            f"score={score} not recognized."
            + f" Must be one of {', '.join(VALID_SCORES)}"
            + " or a tuple of two numba jitted functions."
        )
        raise ValueError(message)
