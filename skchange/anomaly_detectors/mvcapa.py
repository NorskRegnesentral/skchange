"""The subset multivariate collective and point anomalies (MVCAPA) algorithm."""

__author__ = ["Tveten"]
__all__ = ["Mvcapa"]

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from skchange.utils.numba.njit import njit
from scipy.stats import chi2

from skchange.anomaly_detectors.base import SubsetCollectiveAnomalyDetector
from skchange.costs.saving_factory import saving_factory
from skchange.utils.validation.data import check_data
from skchange.utils.validation.parameters import check_larger_than


def dense_capa_penalty(
    n: int, p: int, n_params: int = 1, scale: float = 1.0
) -> tuple[float, np.ndarray]:
    """Penalty function for dense anomalies in CAPA.

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Dimension of the data.
    n_params : int, optional (default=1)
        Number of parameters per segment in the model.
    scale : float, optional (default=1.0)
        Scaling factor for the penalty components.

    Returns
    -------
    alpha : float
        Constant/global penalty term.
    betas : np.ndarray
        Per-component penalty terms.
    """
    psi = np.log(n)
    penalty = scale * (p * n_params + 2 * np.sqrt(p * n_params * psi) + 2 * psi)
    return penalty, np.zeros(p)


def sparse_capa_penalty(
    n: int, p: int, n_params: int = 1, scale: float = 1.0
) -> tuple[float, np.ndarray]:
    """Penalty function for sparse anomalies in CAPA.

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Dimension of the data.
    n_params : int, optional (default=1)
        Number of parameters per segment in the model.
    scale : float, optional (default=1.0)
        Scaling factor for the penalty components.

    Returns
    -------
    alpha : float
        Constant/global penalty term.
    betas : np.ndarray
        Per-component penalty terms.
    """
    psi = np.log(n)
    dense_penalty = 2 * scale * psi
    sparse_penalty = 2 * scale * np.log(n_params * p)
    return dense_penalty, np.full(p, sparse_penalty)


def intermediate_capa_penalty(
    n: int, p: int, n_params: int = 1, scale: float = 1.0
) -> tuple[float, np.ndarray]:
    """Penalty function balancing both dense and sparse anomalies in CAPA.

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Dimension of the data.
    n_params : int, optional (default=1)
        Number of parameters per segment in the model.
    scale : float, optional (default=1.0)
        Scaling factor for the penalty components.

    Returns
    -------
    alpha : float
        Constant/global penalty term.
    betas : np.ndarray
        Per-component penalty terms.
    """
    if p < 2:
        raise ValueError("p must be at least 2.")

    def penalty_func(j: int) -> float:
        psi = np.log(n)
        c_j = chi2.ppf(1 - j / p, n_params)
        f_j = chi2.pdf(c_j, n_params)
        return scale * (
            2 * (psi + np.log(p))
            + j * n_params
            + 2 * p * c_j * f_j
            + 2 * np.sqrt((j * n_params + 2 * p * c_j * f_j) * (psi + np.log(p)))
        )

    # Penalty function is not defined for j = p.
    penalties = np.vectorize(penalty_func)(np.arange(1, p))
    return 0.0, np.diff(penalties, prepend=0.0, append=penalties[-1])


def combined_capa_penalty(
    n: int, p: int, n_params: int = 1, scale: float = 1.0
) -> tuple[float, np.ndarray]:
    """Pointwise minimum of dense, sparse and intermediate penalties in CAPA.

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Dimension of the data.
    n_params : int, optional (default=1)
        Number of parameters per segment in the model.
    scale : float, optional (default=1.0)
        Scaling factor for the penalty components.

    Returns
    -------
    alpha : float
        Constant/global penalty term.
    betas : np.ndarray
        Per-component penalty terms.
    """
    if p < 2:
        return dense_capa_penalty(n, 1, n_params, scale)

    dense_alpha, dense_betas = dense_capa_penalty(n, p, n_params, scale)
    sparse_alpha, sparse_betas = sparse_capa_penalty(n, p, n_params, scale)
    intermediate_alpha, intermediate_betas = intermediate_capa_penalty(
        n, p, n_params, scale
    )
    dense_penalties = dense_alpha + np.cumsum(dense_betas)
    sparse_penalties = sparse_alpha + np.cumsum(sparse_betas)
    intermediate_penalties = intermediate_alpha + np.cumsum(intermediate_betas)
    pointwise_min_penalties = np.zeros(p + 1)
    pointwise_min_penalties[1:] = np.minimum(
        dense_penalties, np.minimum(sparse_penalties, intermediate_penalties)
    )
    return 0.0, np.diff(pointwise_min_penalties)


def capa_penalty_factory(penalty: Union[str, Callable] = "combined") -> Callable:
    """Get a CAPA penalty function.

    Parameters
    ----------
    penalty : str or Callable, optional (default="combined")
        Penalty function to use for CAPA. If a string, must be one of "dense",
        "sparse", "intermediate" or "combined". If a Callable, must be a function
        returning a penalty and per-component penalties, given n, p, n_params and scale.

    Returns
    -------
    penalty_func : Callable
        Penalty function.
    """
    if callable(penalty):
        return penalty
    elif penalty == "dense":
        return dense_capa_penalty
    elif penalty == "sparse":
        return sparse_capa_penalty
    elif penalty == "intermediate":
        return intermediate_capa_penalty
    elif penalty == "combined":
        return combined_capa_penalty
    else:
        raise ValueError(f"Unknown penalty: {penalty}")


@njit
def get_anomalies(
    anomaly_starts: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    collective_anomalies = []
    point_anomalies = []
    i = anomaly_starts.size - 1
    while i >= 0:
        start_i = anomaly_starts[i]
        size = i - start_i + 1
        if size > 1:
            collective_anomalies.append((int(start_i), i))
            i = int(start_i)
        elif size == 1:
            point_anomalies.append((i, i))
        i -= 1
    return collective_anomalies, point_anomalies


@njit
def penalise_savings(
    savings: np.ndarray, alpha: float, betas: np.ndarray
) -> np.ndarray:
    if np.all(betas < 1e-8):
        penalised_savings = savings.sum(axis=1) - alpha
    if np.all(betas == betas[0]):
        penalised_saving_matrix = np.maximum(savings - betas[0], 0.0) - alpha
        penalised_savings = penalised_saving_matrix.sum(axis=1)
    else:
        n_savings = savings.shape[0]
        penalised_savings = np.zeros(n_savings)
        for i in range(n_savings):
            saving_i = savings[i]
            saving_order = (-saving_i).argsort()  # Decreasing order.
            penalised_saving = np.cumsum(saving_i[saving_order] - betas) - alpha
            argmax = np.argmax(penalised_saving)
            penalised_savings[i] = penalised_saving[argmax]
    return penalised_savings


@njit
def find_affected_components(
    params: Union[np.ndarray, tuple],
    saving_func: Callable,
    anomalies: list[tuple[int, int]],
    alpha: float,
    betas: np.ndarray,
) -> list[tuple[int, int, np.ndarray]]:
    new_anomalies = []
    for start, end in anomalies:
        saving = saving_func(params, np.array([start]), np.array([end]))[0]
        saving_order = (-saving).argsort()  # Decreasing order.
        penalised_saving = np.cumsum(saving[saving_order] - betas) - alpha
        argmax = np.argmax(penalised_saving)
        new_anomalies.append((start, end, saving_order[: argmax + 1]))
    return new_anomalies


@njit
def optimise_savings(
    starts: np.ndarray,
    opt_savings: np.ndarray,
    next_savings: np.ndarray,
    alpha: float,
    betas: np.ndarray,
) -> tuple[float, int, np.ndarray]:
    penalised_saving = penalise_savings(next_savings, alpha, betas)
    candidate_savings = opt_savings[starts] + penalised_saving
    argmax = np.argmax(candidate_savings)
    opt_start = starts[0] + argmax
    return candidate_savings[argmax], opt_start, candidate_savings


@njit
def run_base_capa(
    X: np.ndarray,
    params: Union[np.ndarray, tuple],
    saving_func: Callable,
    collective_alpha: float,
    collective_betas: np.ndarray,
    point_alpha: float,
    point_betas: np.ndarray,
    min_segment_length: int,
    max_segment_length: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    n = X.shape[0]
    opt_savings = np.zeros(n + 1)
    # Store the optimal start and affected components of an anomaly for each t.
    # Used to get the final set of anomalies after the loop.
    opt_anomaly_starts = np.repeat(np.nan, n)
    starts = np.array([0])

    ts = np.arange(min_segment_length - 1, n)
    for t in ts:
        # Collective anomalies
        t_array = np.array([t])
        starts = np.concatenate((starts, t_array - min_segment_length + 1))
        ends = np.repeat(t, len(starts))
        collective_savings = saving_func(params, starts, ends)
        opt_collective_saving, opt_start, candidate_savings = optimise_savings(
            starts, opt_savings, collective_savings, collective_alpha, collective_betas
        )

        # Point anomalies
        point_savings = saving_func(params, t_array, t_array)
        opt_point_saving, _, _ = optimise_savings(
            t_array, opt_savings, point_savings, point_alpha, point_betas
        )

        # Combine and store results
        savings = np.array([opt_savings[t], opt_collective_saving, opt_point_saving])
        argmax = np.argmax(savings)
        opt_savings[t + 1] = savings[argmax]
        if argmax == 1:
            opt_anomaly_starts[t] = opt_start
        elif argmax == 2:
            opt_anomaly_starts[t] = t

        # Pruning the admissible starts
        penalty_sum = collective_alpha + collective_betas.sum()
        saving_too_low = candidate_savings + penalty_sum < opt_savings[t + 1]
        too_long_segment = starts < t - max_segment_length + 2
        prune = saving_too_low | too_long_segment
        starts = starts[~prune]

    collective_anomalies, point_anomalies = get_anomalies(opt_anomaly_starts)
    return opt_savings[1:], collective_anomalies, point_anomalies


@njit
def run_mvcapa(
    X: np.ndarray,
    saving_func: Callable,
    saving_init_func: Callable,
    collective_alpha: float,
    collective_betas: np.ndarray,
    point_alpha: float,
    point_betas: np.ndarray,
    min_segment_length: int,
    max_segment_length: int,
) -> tuple[
    np.ndarray, list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]
]:
    params = saving_init_func(X)
    opt_savings, collective_anomalies, point_anomalies = run_base_capa(
        X,
        params,
        saving_func,
        collective_alpha,
        collective_betas,
        point_alpha,
        point_betas,
        min_segment_length,
        max_segment_length,
    )
    collective_anomalies = find_affected_components(
        params,
        saving_func,
        collective_anomalies,
        collective_alpha,
        collective_betas,
    )
    point_anomalies = find_affected_components(
        params, saving_func, point_anomalies, point_alpha, point_betas
    )
    return opt_savings, collective_anomalies, point_anomalies


class Mvcapa(SubsetCollectiveAnomalyDetector):
    """Subset multivariate collective and point anomaly detection.

    An efficient implementation of the MVCAPA algorithm [1]_ for anomaly detection.

    Parameters
    ----------
    saving : str, default="mean"
        Saving function to use for anomaly detection.
    collective_penalty : str or Callable, optional, default="combined"
        Penalty function to use for collective anomalies. If a string, must be one of
        "dense", "sparse", "intermediate" or "combined". If a Callable, must be a
        function returning a penalty and per-component penalties, given n, p, n_params
        and scale.
    collective_penalty_scale : float, optional, default=1.0
        Scaling factor for the collective penalty.
    point_penalty : str or Callable, optional, default="sparse"
        Penalty function to use for point anomalies. See `collective_penalty`.
    point_penalty_scale : float, optional, default=1.0
        Scaling factor for the point penalty.
    min_segment_length : int, optional, default=2
        Minimum length of a segment.
    max_segment_length : int, optional, default=1000
        Maximum length of a segment.
    ignore_point_anomalies : bool, optional, default=False
        If True, detected point anomalies are not returned by `predict`. I.e., only
        collective anomalies are returned.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       collective and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.anomaly_detectors import Mvcapa
    >>> from skchange.datasets.generate import generate_anomalous_data
    >>> n = 300
    >>> means = [np.array([8.0, 0.0, 0.0]), np.array([2.0, 3.0, 5.0])]
    >>> df = generate_anomalous_data(
    >>>     n, anomalies=[(100, 119), (250, 299)], means=means, random_state=3
    >>> )
    >>> detector = Mvcapa()
    >>> detector.fit_predict(df)
      anomaly_interval anomaly_columns
    0       [100, 119]             [0]
    1       [250, 299]       [2, 1, 0]
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        saving: Union[str, tuple[Callable, Callable]] = "mean",
        collective_penalty: Union[str, Callable] = "combined",
        collective_penalty_scale: float = 2.0,
        point_penalty: Union[str, Callable] = "sparse",
        point_penalty_scale: float = 2.0,
        min_segment_length: int = 2,
        max_segment_length: int = 1000,
        ignore_point_anomalies: bool = False,
    ):
        self.saving = saving
        self.collective_penalty = collective_penalty
        self.collective_penalty_scale = collective_penalty_scale
        self.point_penalty = point_penalty
        self.point_penalty_scale = point_penalty_scale
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.ignore_point_anomalies = ignore_point_anomalies
        super().__init__()

        self.saving_func, self.saving_init_func = saving_factory(self.saving)

        check_larger_than(0, collective_penalty_scale, "collective_penalty_scale")
        check_larger_than(0, point_penalty_scale, "point_penalty_scale")
        check_larger_than(2, min_segment_length, "min_segment_length")
        check_larger_than(min_segment_length, max_segment_length, "max_segment_length")

    def _get_penalty_components(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        # TODO: Add penalty tuning.
        # if self.tune:
        #     return self._tune_threshold(X)
        n = X.shape[0]
        p = X.shape[1]
        n_params = 1  # TODO: Add support for depending on 'score'.
        collective_penalty_func = capa_penalty_factory(self.collective_penalty)
        collective_alpha, collective_betas = collective_penalty_func(
            n, p, n_params, scale=self.collective_penalty_scale
        )
        point_penalty_func = capa_penalty_factory(self.point_penalty)
        point_alpha, point_betas = point_penalty_func(
            n, p, n_params, scale=self.point_penalty_scale
        )
        return collective_alpha, collective_betas, point_alpha, point_betas

    def _fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """Fit to training data.

        Sets the penalty of the detector.
        If `penalty_scale` is None, the penalty is set to the (1-`level`)-quantile
        of the change/anomaly scores on the training data. For this to be correct,
        the training data must contain no changepoints. If `penalty_scale` is a
        number, the penalty is set to `penalty_scale` times the default penalty
        for the detector. The default penalty depends at least on the data's shape,
        but could also depend on more parameters.

        Parameters
        ----------
        X : pd.DataFrame
            training data to fit the threshold to.
        y : pd.Series, optional
            Does nothing. Only here to make the fit method compatible with sktime
            and scikit-learn.

        Returns
        -------
        self : returns a reference to self

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """
        X = check_data(
            X,
            min_length=self.min_segment_length,
            min_length_name="min_segment_length",
        )
        penalty_components = self._get_penalty_components(X)
        self.collective_alpha_ = penalty_components[0]
        self.collective_betas_ = penalty_components[1]
        self.point_alpha_ = penalty_components[2]
        self.point_betas_ = penalty_components[3]
        return self

    def _predict(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame - data to annotate, time series

        Returns
        -------
        y : pd.Series or pd.DataFrame
            Annotations for sequence X, exact format depends on annotation type.
        """
        X = check_data(
            X,
            min_length=self.min_segment_length,
            min_length_name="min_segment_length",
        )
        opt_savings, collective_anomalies, point_anomalies = run_mvcapa(
            X.values,
            self.saving_func,
            self.saving_init_func,
            self.collective_alpha_,
            self.collective_betas_,
            self.point_alpha_,
            self.point_betas_,
            self.min_segment_length,
            self.max_segment_length,
        )
        self.scores = pd.Series(opt_savings, index=X.index, name="score")

        anomalies = collective_anomalies
        if not self.ignore_point_anomalies:
            anomalies += point_anomalies
        anomalies = sorted(anomalies)

        return SubsetCollectiveAnomalyDetector._format_sparse_output(anomalies)

    def _score_transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Compute the MVCAPA scores for the input data.

        Parameters
        ----------
        X : pd.DataFrame - data to compute scores for, time series

        Returns
        -------
        scores : pd.Series - scores for sequence X

        Notes
        -----
        The MVCAPA scores are the cumulative optimal savings, so the scores are
        increasing and are not per observation scores.
        """
        self.predict(X)
        return self.scores

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for annotators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            {"saving": "mean", "min_segment_length": 5, "max_segment_length": 100},
            {"saving": "mean", "min_segment_length": 2, "max_segment_length": 20},
        ]
        return params
