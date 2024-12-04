"""The subset multivariate collective and point anomalies (MVCAPA) algorithm."""

__author__ = ["Tveten"]
__all__ = ["MVCAPA"]

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2

from skchange.anomaly_detectors.base import SubsetCollectiveAnomalyDetector
from skchange.anomaly_scores import BaseSaving, L2Saving, to_saving
from skchange.costs import BaseCost
from skchange.utils.numba import njit
from skchange.utils.validation.data import check_data
from skchange.utils.validation.parameters import check_larger_than


def capa_penalty(n: int, n_params: int = 1, scale: float = 1.0) -> float:
    """Get the default penalty for CAPA.

    Parameters
    ----------
    n : int
        Sample size.
    n_params: int, optional (default=1)
        Number of parameters per segment in the model across all variables.
    scale : float, optional (default=1.0)
        Scaling factor for the penalty components.

    Returns
    -------
    penalty : float
        Penalty value.
    """
    psi = np.log(n)
    penalty = scale * (n_params + 2 * np.sqrt(n_params * psi) + 2 * psi)
    return penalty


def dense_mvcapa_penalty(
    n: int, p: int, n_params_per_variable: int = 1, scale: float = 1.0
) -> tuple[float, np.ndarray]:
    """Penalty function for dense anomalies in CAPA.

    Parameters
    ----------
    n : int
        Sample size.
    n_params: int, optional (default=1)
        Number of parameters per segment in the model across all variables.
    scale : float, optional (default=1.0)
        Scaling factor for the penalty components.

    Returns
    -------
    alpha : float
        Constant/global penalty term.
    betas : np.ndarray
        Per-component penalty terms.
    """
    return capa_penalty(n, p * n_params_per_variable, scale), np.zeros(p)


def sparse_mvcapa_penalty(
    n: int, p: int, n_params_per_variable: int = 1, scale: float = 1.0
) -> tuple[float, np.ndarray]:
    """Penalty function for sparse anomalies in CAPA.

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Dimension of the data.
    n_params_per_variable : int, optional (default=1)
        Number of parameters per variable and segment in the model.
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
    sparse_penalty = 2 * scale * np.log(n_params_per_variable * p)
    return dense_penalty, np.full(p, sparse_penalty)


def intermediate_mvcapa_penalty(
    n: int, p: int, n_params_per_variable: int = 1, scale: float = 1.0
) -> tuple[float, np.ndarray]:
    """Penalty function balancing both dense and sparse anomalies in CAPA.

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Dimension of the data.
    n_params_per_variable : int, optional (default=1)
        Number of parameters per variable and segment in the model.
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
        c_j = chi2.ppf(1 - j / p, n_params_per_variable)
        f_j = chi2.pdf(c_j, n_params_per_variable)
        return scale * (
            2 * (psi + np.log(p))
            + j * n_params_per_variable
            + 2 * p * c_j * f_j
            + 2
            * np.sqrt(
                (j * n_params_per_variable + 2 * p * c_j * f_j) * (psi + np.log(p))
            )
        )

    # Penalty function is not defined for j = p.
    penalties = np.vectorize(penalty_func)(np.arange(1, p))
    return 0.0, np.diff(penalties, prepend=0.0, append=penalties[-1])


def combined_mvcapa_penalty(
    n: int, p: int, n_params_per_variable: int = 1, scale: float = 1.0
) -> tuple[float, np.ndarray]:
    """Pointwise minimum of dense, sparse and intermediate penalties in CAPA.

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Dimension of the data.
    n_params_per_variable : int, optional (default=1)
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
        return dense_mvcapa_penalty(n, 1, n_params_per_variable, scale)

    dense_alpha, dense_betas = dense_mvcapa_penalty(n, p * n_params_per_variable, scale)
    dense_betas = np.zeros(p)
    sparse_alpha, sparse_betas = sparse_mvcapa_penalty(
        n, p, n_params_per_variable, scale
    )
    intermediate_alpha, intermediate_betas = intermediate_mvcapa_penalty(
        n, p, n_params_per_variable, scale
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
        return dense_mvcapa_penalty
    elif penalty == "sparse":
        return sparse_mvcapa_penalty
    elif penalty == "intermediate":
        return intermediate_mvcapa_penalty
    elif penalty == "combined":
        return combined_mvcapa_penalty
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
            collective_anomalies.append((int(start_i), i + 1))
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


def find_affected_components(
    saving: BaseSaving,
    anomalies: list[tuple[int, int]],
    alpha: float,
    betas: np.ndarray,
) -> list[tuple[int, int, np.ndarray]]:
    saving.check_is_fitted()
    new_anomalies = []
    for start, end in anomalies:
        saving_values = saving.evaluate(np.array([start, end]))[0]
        saving_order = (-saving_values).argsort()  # Decreasing order.
        penalised_saving = np.cumsum(saving_values[saving_order] - betas) - alpha
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


def run_base_capa(
    collective_saving: BaseSaving,
    point_saving: BaseSaving,
    collective_alpha: float,
    collective_betas: np.ndarray,
    point_alpha: float,
    point_betas: np.ndarray,
    min_segment_length: int,
    max_segment_length: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    collective_saving.check_is_fitted()
    point_saving.check_is_fitted()

    n = collective_saving._X.shape[0]
    opt_savings = np.zeros(n + 1)
    # Store the optimal start and affected components of an anomaly for each t.
    # Used to get the final set of anomalies after the loop.
    opt_anomaly_starts = np.repeat(np.nan, n)
    starts = np.array([], dtype=int)

    ts = np.arange(min_segment_length - 1, n)
    for t in ts:
        # Collective anomalies
        t_array = np.array([t])
        starts = np.concatenate((starts, t_array - min_segment_length + 1))
        ends = np.repeat(t + 1, len(starts))
        collective_savings = collective_saving.evaluate(np.column_stack((starts, ends)))
        opt_collective_saving, opt_start, candidate_savings = optimise_savings(
            starts, opt_savings, collective_savings, collective_alpha, collective_betas
        )

        # Point anomalies
        point_savings = point_saving.evaluate(np.column_stack((t_array, t_array + 1)))
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


def run_mvcapa(
    X: np.ndarray,
    collective_saving: BaseSaving,
    point_saving: BaseSaving,
    collective_penalty: str,
    collective_penalty_scale: float,
    point_penalty: str,
    point_penalty_scale: float,
    min_segment_length: int,
    max_segment_length: int,
) -> tuple[
    np.ndarray, list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]
]:
    n = X.shape[0]
    p = X.shape[1]
    collective_n_params_per_variable = collective_saving.get_param_size(1)
    collective_penalty_func = capa_penalty_factory(collective_penalty)
    collective_alpha, collective_betas = collective_penalty_func(
        n, p, collective_n_params_per_variable, scale=collective_penalty_scale
    )
    point_penalty_func = capa_penalty_factory(point_penalty)
    point_n_params_per_variable = point_saving.get_param_size(1)
    point_alpha, point_betas = point_penalty_func(
        n, p, point_n_params_per_variable, scale=point_penalty_scale
    )
    collective_saving.fit(X)
    point_saving.fit(X)
    opt_savings, collective_anomalies, point_anomalies = run_base_capa(
        collective_saving,
        point_saving,
        collective_alpha,
        collective_betas,
        point_alpha,
        point_betas,
        min_segment_length,
        max_segment_length,
    )

    sparse_penalty_func = capa_penalty_factory("sparse")
    sparse_alpha, sparse_betas = sparse_penalty_func(
        n, p, collective_n_params_per_variable, scale=collective_penalty_scale
    )
    collective_anomalies = find_affected_components(
        collective_saving,
        collective_anomalies,
        sparse_alpha,
        sparse_betas,
    )
    point_anomalies = find_affected_components(
        point_saving, point_anomalies, point_alpha, point_betas
    )
    return opt_savings, collective_anomalies, point_anomalies


class MVCAPA(SubsetCollectiveAnomalyDetector):
    """Subset multivariate collective and point anomaly detection.

    An efficient implementation of the MVCAPA algorithm [1]_ for anomaly detection.

    Parameters
    ----------
    collective_saving : BaseSaving or BaseCost, optional, default=L2Saving()
        The saving function to use for collective anomaly detection.
        Only univariate savings are permitted (see the `evaluation_type` attribute).
        If a `BaseCost` is given, the saving function is constructed from the cost. The
        cost must have a fixed parameter that represents the baseline cost.
    point_saving : BaseSaving or BaseCost, optional, default=L2Saving()
        The saving function to use for point anomaly detection. Only savings with a
        minimum size of 1 are permitted.
        If a `BaseCost` is given, the saving function is constructed from the cost. The
        cost must have a fixed parameter that represents the baseline cost.
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
    >>> from skchange.anomaly_detectors import MVCAPA
    >>> from skchange.datasets.generate import generate_anomalous_data
    >>> n = 300
    >>> means = [np.array([8.0, 0.0, 0.0]), np.array([2.0, 3.0, 5.0])]
    >>> df = generate_anomalous_data(
    >>>     n, anomalies=[(100, 120), (250, 300)], means=means, random_state=3
    >>> )
    >>> detector = MVCAPA()
    >>> detector.fit_predict(df)
      anomaly_interval anomaly_columns
    0       [100, 120)             [0]
    1       [250, 300)       [2, 1, 0]

    Notes
    -----
    The MVCAPA algorithm assumes the input data is centered before fitting and
    predicting.
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        collective_saving: Optional[Union[BaseSaving, BaseCost]] = None,
        point_saving: Optional[Union[BaseSaving, BaseCost]] = None,
        collective_penalty: Union[str, Callable] = "combined",
        collective_penalty_scale: float = 2.0,
        point_penalty: Union[str, Callable] = "sparse",
        point_penalty_scale: float = 2.0,
        min_segment_length: int = 2,
        max_segment_length: int = 1000,
        ignore_point_anomalies: bool = False,
    ):
        self.collective_saving = collective_saving
        self.point_saving = point_saving
        self.collective_penalty = collective_penalty
        self.collective_penalty_scale = collective_penalty_scale
        self.point_penalty = point_penalty
        self.point_penalty_scale = point_penalty_scale
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.ignore_point_anomalies = ignore_point_anomalies
        super().__init__()

        _collective_saving = (
            L2Saving() if collective_saving is None else collective_saving
        )
        if _collective_saving.evaluation_type == "multivariate":
            raise ValueError("Collective saving must be univariate.")
        self._collective_saving = to_saving(_collective_saving)

        _point_saving = L2Saving() if point_saving is None else point_saving
        if _point_saving.min_size != 1:
            raise ValueError("Point saving must have a minimum size of 1.")
        self._point_saving = to_saving(_point_saving)

        check_larger_than(0, collective_penalty_scale, "collective_penalty_scale")
        check_larger_than(0, point_penalty_scale, "point_penalty_scale")
        check_larger_than(2, min_segment_length, "min_segment_length")
        check_larger_than(min_segment_length, max_segment_length, "max_segment_length")

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
            Training data to fit the threshold to.
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
        return self

    def _predict(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Detect events and return the result in a sparse format.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame or np.ndarray
            Data to detect events in (time series).

        Returns
        -------
        y : pd.Series or pd.DataFrame
            Each element or row corresponds to a detected event. Exact format depends on
            the detector type.
        """
        X = check_data(
            X,
            min_length=self.min_segment_length,
            min_length_name="min_segment_length",
        )
        opt_savings, collective_anomalies, point_anomalies = run_mvcapa(
            X.values,
            self._collective_saving,
            self._point_saving,
            self.collective_penalty,
            self.collective_penalty_scale,
            self.point_penalty,
            self.point_penalty_scale,
            self.min_segment_length,
            self.max_segment_length,
        )
        self.scores = pd.Series(opt_savings, index=X.index, name="score")

        anomalies = collective_anomalies
        if not self.ignore_point_anomalies:
            anomalies += point_anomalies
        anomalies = sorted(anomalies)

        return SubsetCollectiveAnomalyDetector._format_sparse_output(anomalies)

    def _transform_scores(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
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
        from skchange.costs import L2Cost

        params = [
            {
                "collective_saving": L2Cost(param=0.0),
                "point_saving": L2Cost(param=0.0),
                "min_segment_length": 5,
                "max_segment_length": 100,
            },
            {
                "collective_saving": L2Cost(param=0.0),
                "point_saving": L2Cost(param=0.0),
                "min_segment_length": 2,
                "max_segment_length": 20,
            },
        ]
        return params
