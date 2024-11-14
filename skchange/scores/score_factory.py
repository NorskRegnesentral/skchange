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

__author__ = ["Tveten"]

from typing import Callable, Union

from skchange.scores.mean_cov_score import (
    init_mean_cov_score,
    mean_cov_score,
)
from skchange.scores.mean_score import init_mean_score, mean_anomaly_score, mean_score
from skchange.scores.mean_var_score import (
    init_mean_var_score,
    mean_var_anomaly_score,
    mean_var_score,
)

VALID_CHANGE_SCORES = ["mean", "mean_var", "mean_cov"]
VALID_ANOMALY_SCORES = ["mean", "mean_var"]


def score_factory(score: Union[str, tuple[Callable, Callable]] = "mean"):
    """Return score function and its initializer.

    Parameters
    ----------
    score : {"mean", "mean_var", "mean_cov"}, tuple[Callable, Callable], default="mean"
        Test statistic to use for changepoint detection.

        * `"mean"`: The CUSUM statistic for a change in mean (this is equivalent to a
          likelihood ratio test for a change in the mean of Gaussian data). For
          multivariate data, the sum of the CUSUM statistics for each dimension is used.
        * `"mean_var"`: The likelihood ratio test for a change in the mean and/or
          variance of Gaussian data. For multivariate data, the sum of the likelihood
          ratio statistics for each dimension is used.
        * `"mean_cov"`: The likelihood ratio test for a change in the mean and/or
          covariance matrix of multivariate Gaussian data.
        * If a tuple, it must contain two numba jitted functions:

            1. The first function is the scoring function, which takes four arguments:

                1. `precomputed_params`: The output of the second function.
                2. `starts`: Start indices of the intervals to score for a change.
                3. `ends`: End indices of the intervals to score for a change.
                4. `splits`: Split indices of the intervals to score for a change.

               For each start, split and end, the score should be calculated for the
               data intervals `[start:split]` and `[split+1:end]`, meaning that both
               the starts and ends are inclusive, while split is included in the left
               interval.

            2. The second function is the initializer, which takes the data matrix as
               input and returns precomputed quantities that may speed up the score
               calculations. If not relevant, just return the data matrix.

    Returns
    -------
    score_func : Numba jitted Callable
        Score function.
    init_score_func : Numba jitted Callable
        Score function initializer.

    Raises
    ------
    ValueError
        If the provided `score` is not recognized, an error is raised with a message
        indicating the valid options.
    """
    if isinstance(score, str) and score == "mean":
        return mean_score, init_mean_score
    elif isinstance(score, str) and score == "mean_var":
        return mean_var_score, init_mean_var_score
    elif isinstance(score, str) and score == "mean_cov":
        return mean_cov_score, init_mean_cov_score
    elif len(score) == 2:
        return score[0], score[1]
    else:
        message = (
            f"score={score} not recognized."
            + f" Must be one of {', '.join(VALID_CHANGE_SCORES)}"
            + " or a tuple of two numba jitted functions."
        )
        raise ValueError(message)


def anomaly_score_factory(score: Union[str, tuple[Callable, Callable]] = "mean"):
    """Return anomaly score function and its initializer.

    Parameters
    ----------
    score: {"mean", "mean_var"}, tuple[Callable, Callable], default="mean"
        Test statistic to use for anomaly detection.

        * `"mean"`: The CUSUM statistic for a change in mean (this is equivalent to a
          likelihood ratio test for a change in the mean of Gaussian data). For
          multivariate data, the sum of the CUSUM statistics for each dimension is used.
        * `"mean_var"`: The likelihood ratio test for a change in the mean and/or
          variance of Gaussian data. For multivariate data, the sum of the likelihood
          ratio statistics for each dimension is used.
        * If a tuple, it must contain two numba jitted functions:

            1. The first function is the scoring function, which takes five arguments:

                1. `precomputed_params`: The output of the second function.
                2. `interval_starts`: Start indices of the intervals to to test for
                anomalies within.
                3. `interval_ends`: End indices of the intervals to to test for
                anomalies within.
                4. `anomaly_starts`: Start indices of the anomalies.
                5. `anomaly_ends`: End indices of the anomalies.

            For each interval_start, interval_end, anomaly_start and anomaly_end, the
            score should compare the data in `[anomaly_start:anomaly_end+1]` to the
            data in `[interval_start:interval_end+1]` that is not in
            `[anomaly_start:anomaly_end+1]`.

            The overall and anomalous intervals must
            satisfy `interval_start > anomaly_start <= anomaly_end <= interval_end`.

            2. The second function is the initializer, which takes the data matrix as
               input and returns precomputed quantities that may speed up the score
               calculations. If not relevant, just return the data matrix.

    Returns
    -------
    score_func : Numba jitted Callable
        Score function.
    init_score_func : Numba jitted Callable
        Score function initializer.
    """
    if isinstance(score, str) and score == "mean":
        return mean_anomaly_score, init_mean_score
    elif isinstance(score, str) and score == "mean_var":
        return mean_var_anomaly_score, init_mean_var_score
    elif len(score) == 2 and all([is_jitted(s) for s in score]):
        return score[0], score[1]
    else:
        message = (
            f"score={score} not recognized."
            + f" Must be one of {', '.join(VALID_ANOMALY_SCORES)}"
            + " or a tuple of two numba jitted functions."
        )
        raise ValueError(message)
