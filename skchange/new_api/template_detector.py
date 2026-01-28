"""Template for implementing custom change detectors.

This template demonstrates how to create a sklearn-compatible change detector
using the new skchange API. Copy this file and follow the TODO comments.

Author: Your Name
License: BSD 3-Clause
"""

from __future__ import annotations

import numpy as np

from skchange.new_api.base import BaseChangeDetector
from skchange.new_api.typing import ArrayLike, Segmentation
from skchange.new_api.utils import make_segmentation, validate_data


# TODO: Rename this class to your detector name (e.g., MyPELT, MyBinSeg)
class TemplateDetector(BaseChangeDetector):
    """Template for a change detection algorithm.

    TODO: Write a brief description of your algorithm here.
    Explain what makes it unique or when it should be used.

    Parameters
    ----------
    threshold : float, default=1.0
        TODO: Document your primary hyperparameter.
        Explain what it controls and how to tune it.

    min_segment_length : int, default=2
        TODO: Add more parameters as needed.
        Each parameter should have a clear description.

    Attributes
    ----------
    threshold_ : float
        TODO: Document attributes learned during fit().
        All fitted attributes must end with underscore.

    changepoints_ : np.ndarray
        TODO: Example of storing last prediction (optional).
        Only if your algorithm needs to remember state.

    Examples
    --------
    >>> from skchange.new_api.template_detector import TemplateDetector
    >>> import numpy as np
    >>>
    >>> # Generate example data
    >>> X = np.concatenate([
    ...     np.random.randn(50, 1),
    ...     np.random.randn(50, 1) + 2,
    ...     np.random.randn(50, 1)
    ... ])
    >>>
    >>> # Fit and predict
    >>> detector = TemplateDetector(threshold=1.5)
    >>> detector.fit(X)
    >>> result = detector.predict(X)
    >>> print(result["changepoints"])
    [50 100]
    >>>
    >>> # Get dense labels
    >>> labels = detector.transform(X)
    >>> print(labels.shape)
    (150,)

    Notes
    -----
    TODO: Add implementation notes, algorithm references, or complexity.

    References
    ----------
    .. [1] TODO: Add key paper references for your algorithm.
           Author et al. "Paper Title." Journal, Year.

    See Also
    --------
    TODO: Link to related detectors or base classes.
    """

    # TODO: Add all hyperparameters as keyword arguments with defaults
    def __init__(
        self,
        threshold: float = 1.0,
        min_segment_length: int = 2,
    ):
        # Store all parameters as attributes (required by sklearn)
        # Parameter names MUST match __init__ arguments exactly
        self.threshold = threshold
        self.min_segment_length = min_segment_length

        # NOTE: Do NOT do validation or computation in __init__
        # All logic should go in fit() or predict()

    def fit(self, X: ArrayLike, y: Segmentation | ArrayLike | None = None):
        """Fit the change detector on training data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training time series data.
            - Univariate: shape (n_samples, 1)
            - Multivariate: shape (n_samples, n_features)

        y : Segmentation | ArrayLike | None, default=None
            Optional supervised labels.
            - Segmentation dict: sparse changepoint labels
            - ArrayLike: dense per-sample labels
            - None: unsupervised (most common)

        Returns
        -------
        self : TemplateDetector
            Fitted detector instance.
        """
        # TODO: Validate and convert input data
        # This also sets self.n_features_in_ automatically
        X = validate_data(self, X)

        # TODO: If you support supervised learning, process y here
        if y is not None:
            # Option 1: y is a Segmentation dict
            if isinstance(y, dict):
                from skchange.new_api.utils import sparse_to_dense

                y_dense = sparse_to_dense(y)
                # Use y_dense for supervised learning
            # Option 2: y is already dense labels
            else:
                y_dense = np.asarray(y)
                if len(y_dense) != len(X):
                    raise ValueError("y must have same length as X")

        # TODO: Learn parameters from data
        # All learned attributes MUST end with underscore
        # Example: compute threshold from data statistics
        self.threshold_ = self.threshold * np.std(X)

        # TODO: Add any other learned parameters
        # self.mean_ = np.mean(X, axis=0)
        # self.n_samples_seen_ = len(X)

        # Required: return self for sklearn compatibility
        return self

    def predict(self, X: ArrayLike) -> Segmentation:
        """Detect changepoints in a time series.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Time series to analyze for changepoints.

        Returns
        -------
        result : Segmentation
            Detection result as a dict with required fields:

            - "changepoints": np.ndarray, changepoint indices
            - "labels": np.ndarray, segment labels
            - "n_samples": int, number of samples

            Optional fields:

            - "n_features": int, number of features
            - "scores": np.ndarray, changepoint scores
            - "affected_variables": list[np.ndarray], per-changepoint variable indices
            - "meta": dict, algorithm-specific metadata
        """
        # TODO: Validate input and check if fitted
        # This ensures fit() was called and validates X shape
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        # TODO: Implement your changepoint detection algorithm
        # This is where the core detection logic goes
        changepoints = self._detect_changepoints(X)

        # TODO: Optionally compute scores for each changepoint
        # scores = self._compute_scores(X, changepoints)

        # TODO: For multivariate data, optionally identify affected variables
        # affected_vars = self._identify_affected_variables(X, changepoints)

        # TODO: Create and return Segmentation dict
        # Use make_segmentation() helper for clean syntax
        result = make_segmentation(
            changepoints=changepoints,
            n_samples=len(X),
            n_features=X.shape[1],  # Optional but recommended
            # scores=scores,  # Optional
            # affected_variables=affected_vars,  # Optional
            meta={  # Optional: algorithm-specific info
                "threshold_used": self.threshold_,
                # "n_iterations": n_iter,
            },
        )

        return result

    # ==================== Optional Overrides ====================

    def __sklearn_tags__(self):
        """Customize sklearn tags for this detector.

        TODO: Override only if you need to change default tags.
        Common cases:
        - Set capability_multiple_series = True for multi-series support
        - Set requires_y = True for supervised detectors
        """
        tags = super().__sklearn_tags__()

        # TODO: Uncomment and modify as needed
        # tags.change_detector_tags.capability_multiple_series = True
        # tags.target_tags.required = True  # If supervised learning
        # tags.input_tags.allow_nan = False  # If you don't handle NaN

        return tags

    # def score(self, X: ArrayLike, y: Segmentation) -> float:
    #     """Override score to use a different evaluation metric.
    #
    #     TODO: Only override if you want a different metric than Hausdorff distance.
    #
    #     The default uses negative Hausdorff distance (higher is better).
    #     Common alternatives: covering metric, F1 score, Rand index.
    #     """
    #     from skchange.new_api.metrics import f1_score
    #
    #     y_pred = self.predict(X)
    #     return f1_score(y, y_pred)


# ==================== Example: Multi-Series Detector ====================


class MultiSeriesTemplateDetector(BaseChangeDetector):
    """Template for detectors that support multi-series training.

    TODO: Use this template if your algorithm can learn from multiple series.

    Examples include:
    - Meta-learning from series-level labels
    - Learning shared parameters across series
    - Batch processing for efficiency

    Parameters
    ----------
    threshold : float, default=1.0
        Detection threshold.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def __sklearn_tags__(self):
        """Enable multi-series support."""
        tags = super().__sklearn_tags__()
        # Enable multi-series capability
        tags.change_detector_tags.capability_multiple_series = True
        return tags

    def fit(
        self,
        X: ArrayLike | list[ArrayLike],
        y: Segmentation | list[Segmentation] | ArrayLike | None = None,
    ):
        """Fit on single or multiple series.

        Parameters
        ----------
        X : ArrayLike | list[ArrayLike]
            Training data.
            - Single series: shape (n_samples, n_features)
            - Multiple series: list of arrays, each (n_samples_i, n_features)

        y : Segmentation | list[Segmentation] | ArrayLike | None
            Labels (optional).
            - For multi-series: list[Segmentation] or array of series labels
            - For single series: Segmentation dict or dense labels

        Returns
        -------
        self
        """
        # TODO: Handle both single and multiple series
        if isinstance(X, list):
            # Multiple series
            X_validated = [
                validate_data(self, X_i, reset=(i == 0)) for i, X_i in enumerate(X)
            ]

            # TODO: Process y if provided
            if y is not None:
                if isinstance(y, list):
                    # Per-series segment labels
                    # Each y_i should be a Segmentation dict
                    pass
                else:
                    # Series-level classification labels
                    # y is array of shape (n_series,)
                    y = np.asarray(y)

            # TODO: Learn shared parameters across all series
            # Example: compute global threshold
            all_data = np.concatenate(X_validated, axis=0)
            self.threshold_ = self.threshold * np.std(all_data)

        else:
            # Single series
            X = validate_data(self, X)
            self.threshold_ = self.threshold * np.std(X)

        return self

    def predict(self, X: ArrayLike) -> Segmentation:
        """Predict on a single series.

        TODO: Implement prediction logic using learned parameters.

        Note: predict() always takes single series, even if fit() saw multiple.
        """
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        # TODO: Use learned threshold_ for detection
        changepoints = np.array([])  # Placeholder

        return make_segmentation(changepoints=changepoints, n_samples=len(X))


# ==================== Testing Your Detector ====================

if __name__ == "__main__":
    """Quick test to verify your detector works."""

    # Generate test data with changepoints at 50 and 100
    np.random.seed(42)
    X_test = np.concatenate(
        [
            np.random.randn(50, 1) + 0,
            np.random.randn(50, 1) + 3,
            np.random.randn(50, 1) + 0,
        ]
    )

    print("Testing TemplateDetector...")
    print("=" * 50)

    # Test basic fit/predict
    detector = TemplateDetector(threshold=1.0)
    detector.fit(X_test)
    result = detector.predict(X_test)

    print(f"Detected {len(result['changepoints'])} changepoints:")
    print(f"  Locations: {result['changepoints']}")
    print(f"  Labels: {result['labels']}")
    print(f"  n_samples: {result['n_samples']}")

    # Test transform
    labels = detector.transform(X_test)
    print(f"\nDense labels shape: {labels.shape}")
    print(f"  Unique labels: {np.unique(labels)}")

    # Test fit_transform
    labels2 = detector.fit_transform(X_test)
    print(f"\nfit_transform result: {labels2.shape}")

    # TODO: Test your detector with sklearn tools
    # from sklearn.model_selection import GridSearchCV
    # param_grid = {"threshold": [0.5, 1.0, 2.0]}
    # grid = GridSearchCV(TemplateDetector(), param_grid, cv=...)
    # grid.fit(X_train, y_train)

    print("\n✓ Basic tests passed!")
    print("\nTODO: Implement your detection algorithm in _detect_changepoints()")
