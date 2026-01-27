"""Model selection and cross-validation for changepoint detection.

This module provides custom cross-validation utilities that understand
the series-level structure of changepoint detection data.

Unlike standard sklearn CV which operates on row samples, our CV treats
each time series as a sample unit. This is necessary because:
1. Series have variable lengths (can't stack into single array)
2. Predictions are series-relative (indices only meaningful within series)
3. Metrics aggregate across series
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.model_selection import BaseCrossValidator, KFold


class SeriesKFold(BaseCrossValidator):
    """K-Fold cross-validator for series data.

    Treats each time series as a single sample for cross-validation.
    Splits series (not timepoints) into train/test folds.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=False
        Whether to shuffle series before splitting.
    random_state : int or None, default=None
        Random state for shuffling.

    Examples
    --------
    >>> cv = SeriesKFold(n_splits=5, shuffle=True, random_state=42)
    >>> for train_idx, test_idx in cv.split(X_list):
    ...     X_train = [X_list[i] for i in train_idx]
    ...     X_test = [X_list[i] for i in test_idx]
    """

    def __init__(
        self, n_splits: int = 5, shuffle: bool = False, random_state: int | None = None
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate train/test splits at series level.

        Parameters
        ----------
        X : list[ArrayLike] or ArrayLike
            Series data. If list, len(X) = n_series.
            If single series, treated as 1 sample (no splits possible).
        y : list or ArrayLike or None
            Labels (not used for splitting logic).
        groups : array-like or None
            Group labels (not used).

        Yields
        ------
        train_idx : np.ndarray
            Indices of training series.
        test_idx : np.ndarray
            Indices of test series.
        """
        n_series = len(X) if isinstance(X, list) else 1

        if n_series < self.n_splits:
            raise ValueError(
                f"Cannot split {n_series} series into {self.n_splits} folds. "
                f"Reduce n_splits or provide more series."
            )

        indices = np.arange(n_series)

        # Use standard KFold on series indices
        kfold = KFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
        )

        yield from kfold.split(indices)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits."""
        return self.n_splits


def cross_val_score(
    estimator,
    X: list,
    y: list | None = None,
    cv: BaseCrossValidator | None = None,
    scoring: Callable | None = None,
) -> np.ndarray:
    """Evaluate estimator using cross-validation on series data.

    Parameters
    ----------
    estimator : estimator object
        Changepoint detector implementing fit() and predict().
    X : list[ArrayLike]
        List of time series data.
    y : list or None, default=None
        Ground truth labels/changepoints for each series.
        Format must match what scoring function expects.
    cv : BaseCrossValidator or None, default=None
        Cross-validation strategy. If None, uses SeriesKFold(5).
    scoring : callable or None, default=None
        Scoring function with signature: scoring(y_true, y_pred) -> float.
        Must handle list inputs (one per series).
        If None, uses estimator's score method (if available).

    Returns
    -------
    np.ndarray
        Array of scores, one per fold.

    Examples
    --------
    >>> from skchange.new_api.metrics import hausdorff_metric
    >>> from skchange.new_api.model_selection import cross_val_score, SeriesKFold
    >>>
    >>> # Prepare data
    >>> X_list = [X1, X2, X3, X4, X5]  # 5 series
    >>> y_true = [np.array([10, 50]), np.array([20]), ...]  # True changepoints
    >>>
    >>> # Cross-validate with changepoint metric
    >>> scores = cross_val_score(
    ...     detector,
    ...     X_list,
    ...     y_true,
    ...     cv=SeriesKFold(n_splits=5),
    ...     scoring=hausdorff_metric
    ... )
    >>> print(f"Mean Hausdorff: {scores.mean():.2f} (+/- {scores.std():.2f})")
    """
    if cv is None:
        cv = SeriesKFold(n_splits=5)

    if not isinstance(X, list):
        raise TypeError("X must be a list of series for cross-validation")

    scores = []

    for train_idx, test_idx in cv.split(X, y):
        # Index into lists
        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]

        if y is not None:
            y_train = [y[i] for i in train_idx]
            y_test = [y[i] for i in test_idx]
        else:
            y_train = None
            y_test = None

        # Fit on training series
        estimator.fit(X_train, y_train)

        # Predict on test series
        y_pred = [estimator.predict(X_test[i]) for i in range(len(X_test))]

        # Compute score
        if scoring is not None:
            score = scoring(y_test, y_pred)
        elif hasattr(estimator, "score"):
            score = estimator.score(X_test, y_test)
        else:
            raise ValueError(
                "No scoring method provided and estimator has no score() method"
            )

        scores.append(score)

    return np.array(scores)


def train_test_split(
    X: list,
    y: list | None = None,
    test_size: float | int | None = None,
    train_size: float | int | None = None,
    random_state: int | None = None,
    shuffle: bool = True,
):
    """Split series data into train and test sets.

    Parameters
    ----------
    X : list[ArrayLike]
        List of time series.
    y : list or None
        Corresponding labels/ground truth.
    test_size : float or int or None
        If float, proportion of series for test (0.0-1.0).
        If int, absolute number of test series.
        If None, complements train_size or defaults to 0.25.
    train_size : float or int or None
        Similar to test_size but for training set.
    random_state : int or None
        Random seed for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle before splitting.

    Returns
    -------
    X_train : list[ArrayLike]
        Training series.
    X_test : list[ArrayLike]
        Test series.
    y_train : list or None
        Training labels (if y provided).
    y_test : list or None
        Test labels (if y provided).

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X_list, y_list, test_size=0.2, random_state=42
    ... )
    """
    if not isinstance(X, list):
        raise TypeError("X must be a list of series")

    n_series = len(X)
    indices = np.arange(n_series)

    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    # Determine split sizes
    if test_size is None and train_size is None:
        test_size = 0.25

    if isinstance(test_size, float):
        n_test = int(n_series * test_size)
    elif isinstance(test_size, int):
        n_test = test_size
    elif test_size is None:
        if isinstance(train_size, float):
            n_test = n_series - int(n_series * train_size)
        else:
            n_test = n_series - train_size
    else:
        raise ValueError("test_size must be float, int, or None")

    # Split indices
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    # Index into lists
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]

    if y is not None:
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test
