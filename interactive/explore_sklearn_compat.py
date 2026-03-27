"""Explore sklearn compatibility of new_api detectors.

Demonstrates:
1. Using detectors in sklearn Pipelines.
2. Using detectors in GroupKFold cross-validation.
3. GridSearchCV over penalty values.
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skchange.new_api.detectors import MovingWindow
from skchange.new_api.metrics import rand_index

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. Pipeline
# ---------------------------------------------------------------------------
print("=== PIPELINE ===")

X = np.vstack([rng.normal(0, 1, (100, 2)), rng.normal(5, 1, (100, 2))])

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("detector", MovingWindow()),
    ]
)
pipe.fit(X)

result = pipe.predict(X)
print("predict() return type:", type(result))
print("predict() shape:", result.shape)

result_cp = pipe[-1].predict_changepoints(pipe[:-1].transform(X))
print("predict_changepoints() changepoints:", result_cp["changepoints"])

labels = pipe.transform(X)
print("transform() shape:", labels.shape)
print()

# ---------------------------------------------------------------------------
# 2. GroupKFold cross-validation
# ---------------------------------------------------------------------------
print("=== GroupKFold / cross_validate ===")

n_series = 100
series_len = 200

X_list, y_list, groups_list = [], [], []
for i in range(n_series):
    X_i = np.vstack(
        [
            rng.normal(0, 1, (series_len // 2, 1)),
            rng.normal(2, 1, (series_len // 2, 1)),
        ]
    )
    # Dense ground-truth segment labels (predict() returns dense array)
    y_i = np.array([0] * (series_len // 2) + [1] * (series_len // 2))
    X_list.append(X_i)
    y_list.append(y_i)
    groups_list.append(np.full(series_len, i))

X_all = np.vstack(X_list)
y_all = np.concatenate(y_list)
groups = np.concatenate(groups_list)


# ---------------------------------------------------------------------------
# 2b. MultiSeriesGridSearchCV
#
# Problem with sklearn's GridSearchCV + GroupKFold:
#   The framework calls estimator.predict(X[test_idx]) with the full concatenated
#   test fold as a single array. A changepoint detector treats this as one long
#   time series, producing monotonically increasing labels across series — even
#   if the split is correct. Series boundaries become false changepoints.
#
# Fix: run fit() + predict() per series within each CV fold.
#   ~40 lines, no sklearn private APIs needed.
# ---------------------------------------------------------------------------


class MultiSeriesGridSearchCV:
    """GridSearchCV that evaluates changepoint detectors per series.

    sklearn's GridSearchCV passes the entire concatenated test fold to
    predict() at once. For changepoint detectors this is wrong: series
    boundaries create artificial changepoints, and segment labels from
    different series are not comparable.

    This class iterates over unique groups in every test fold, running
    fit() + predict() independently per series before aggregating scores.

    For unsupervised detectors (like all skchange detectors) the estimator
    is fit on each test series itself — there is nothing to learn from held-out
    training series. For supervised detectors, swap the inner fit() call to
    use the training fold instead (marked below).

    Parameters
    ----------
    estimator : estimator object
        A skchange (or sklearn) estimator.
    param_grid : dict
        Parameter grid passed to sklearn.model_selection.ParameterGrid.
    cv : CV splitter
        Must yield (train_idx, test_idx) and accept a groups argument,
        e.g. GroupKFold.
    scoring : callable(y_true, y_pred) -> float
        A metric function (not a make_scorer wrapper).
    """

    def __init__(self, estimator, param_grid, cv, scoring):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y, groups):
        param_list = list(ParameterGrid(self.param_grid))
        groups = np.asarray(groups)

        # shape: (n_splits, n_params)
        fold_scores = np.full(
            (self.cv.get_n_splits(X, y, groups), len(param_list)), np.nan
        )

        for fold_idx, (train_idx, test_idx) in enumerate(self.cv.split(X, y, groups)):
            X_test = X[test_idx]
            y_test = y[test_idx]
            groups_test = groups[test_idx]

            # --- uncomment if fitting on training data instead ---
            # X_train = X[train_idx]
            # groups_train = groups[train_idx]

            for param_idx, params in enumerate(param_list):
                est = clone(self.estimator).set_params(**params)
                series_scores = []

                for g in np.unique(groups_test):
                    mask = groups_test == g
                    X_s = X_test[mask]
                    y_s = y_test[mask]

                    # Unsupervised: fit and predict on the same test series.
                    # For supervised detectors, fit on training series instead.
                    est.fit(X_s)
                    y_pred = est.predict(X_s)
                    series_scores.append(self.scoring(y_s, y_pred))

                fold_scores[fold_idx, param_idx] = np.mean(series_scores)

        mean_scores = fold_scores.mean(axis=0)
        std_scores = fold_scores.std(axis=0)
        self.best_index_ = int(np.argmax(mean_scores))
        self.best_params_ = param_list[self.best_index_]
        self.best_score_ = mean_scores[self.best_index_]
        self.cv_results_ = {
            "params": param_list,
            "mean_test_score": mean_scores,
            "std_test_score": std_scores,
            **{f"param_{k}": [p[k] for p in param_list] for k in self.param_grid},
        }
        return self


# ---------------------------------------------------------------------------
# 3. MultiSeriesGridSearchCV over penalty values
# ---------------------------------------------------------------------------
print()
print("=== MultiSeriesGridSearchCV over penalty ===")

param_grid = {"penalty": [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]}

grid = MultiSeriesGridSearchCV(
    MovingWindow(),
    param_grid,
    cv=GroupKFold(n_splits=5),
    scoring=rand_index,
)
grid.fit(X_all, y_all, groups=groups)

for penalty, mean, std in zip(
    grid.cv_results_["param_penalty"],
    grid.cv_results_["mean_test_score"],
    grid.cv_results_["std_test_score"],
):
    print(f"  penalty={penalty:5.1f}  rand index: {mean:.4f} ± {std:.4f}")

print(f"best penalty: {grid.best_params_['penalty']}  (score={grid.best_score_:.4f})")
