"""Factory for getting test statistic functions and their initializers from strings.

Recipe for adding new scores:
    1. Add a new module in skchange.score: <score_name>_score.py
    2. Add two functions:
        init_<score_name>_score, which precomputes quantities that should be
        precomputed, often partial sums and such.
        <score_name>_score, which takes in the output of init_<score_name>_score as its
        first argument, and start, end and split indices as the second, third and
        fourth arguments.
    3. Add the name of the score to VALID_CHANGE_SCORES below.
    4. Add the name of the score to the docstring of score_factory below.
    4. Add a new if-statement in score_factory below.

"""

from typing import Callable, Tuple, Union

from numba.extending import is_jitted

from skchange.scores.mean_score import init_mean_score, mean_anomaly_score, mean_score
from skchange.scores.meanvar_score import init_meanvar_score, meanvar_score

VALID_CHANGE_SCORES = ["mean", "meanvar"]
VALID_ANOMALY_SCORES = ["mean"]


def score_factory(score: Union[str, Tuple[Callable, Callable]]):
    """Return score function and its initializer.

    Parameters
    ----------
    score: str, Tuple[Callable, Callable]
        Test statistic to use for changepoint detection.
        * If "mean", the difference-in-mean statistic is used,
        * If "meanvar", the difference-in-mean-and-variance statistic is used,
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
    elif isinstance(score, str) and score == "meanvar":
        return meanvar_score, init_meanvar_score
    elif len(score) == 2 and all([is_jitted(s) for s in score]):
        return score[0], score[1]
    else:
        message = (
            f"score={score} not recognized."
            + f" Must be one of {', '.join(VALID_CHANGE_SCORES)}"
            + " or a tuple of two numba jitted functions."
        )
        raise ValueError(message)


def anomaly_score_factory(score: Union[str, Tuple[Callable, Callable]]):
    """Return anomaly score function and its initializer.

    Parameters
    ----------
    score: str, Tuple[Callable, Callable]
        Test statistic to use for anomaly detection.
        * If "mean", the difference-in-mean statistic is used,
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
        return mean_anomaly_score, init_mean_score
    elif len(score) == 2 and all([is_jitted(s) for s in score]):
        return score[0], score[1]
    else:
        message = (
            f"score={score} not recognized."
            + f" Must be one of {', '.join(VALID_ANOMALY_SCORES)}"
            + " or a tuple of two numba jitted functions."
        )
        raise ValueError(message)
