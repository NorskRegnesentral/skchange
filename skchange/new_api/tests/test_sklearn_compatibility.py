"""Basic sklearn compatibility tests for estimators in ``skchange.new_api``."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

sklearn = pytest.importorskip("sklearn", minversion="1.6")

from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import parametrize_with_checks

import skchange.new_api as new_api


def _iter_new_api_estimator_classes() -> list[type[BaseEstimator]]:
    """Discover concrete sklearn estimators defined in ``skchange.new_api``.

    Notes
    -----
    - Uses sklearn's ``BaseEstimator`` as the estimator criterion.
    - Excludes test, template, and example modules.
    - Excludes base/abstract and private classes.
    """
    excluded_module_suffixes = (".tests", ".template_detector", ".examples")

    classes: list[type[BaseEstimator]] = []
    seen: set[type[BaseEstimator]] = set()

    for mod_info in pkgutil.walk_packages(new_api.__path__, prefix="skchange.new_api."):
        module_name = mod_info.name

        if module_name.endswith(excluded_module_suffixes):
            continue

        module = importlib.import_module(module_name)

        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, BaseEstimator):
                continue
            if cls is BaseEstimator:
                continue
            if cls in seen:
                continue
            if not cls.__module__.startswith("skchange.new_api"):
                continue
            if cls.__name__.startswith("_"):
                continue
            if cls.__name__.startswith("Base"):
                continue
            if inspect.isabstract(cls):
                continue

            seen.add(cls)
            classes.append(cls)

    classes.sort(key=lambda candidate: (candidate.__module__, candidate.__name__))
    return classes


def _instantiate_estimators(classes: list[type[BaseEstimator]]) -> list[BaseEstimator]:
    """Instantiate estimators with default constructors where possible."""
    estimators: list[BaseEstimator] = []
    for cls in classes:
        init_signature = inspect.signature(cls.__init__)
        has_required_params = any(
            parameter.name != "self"
            and parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            for parameter in init_signature.parameters.values()
        )
        if has_required_params:
            continue

        estimators.append(cls())
    return estimators


NEW_API_ESTIMATORS = _instantiate_estimators(_iter_new_api_estimator_classes())


def _expected_failed_checks(estimator: BaseEstimator) -> dict[str, str]:
    """Return sklearn checks expected to fail for known API differences.

    Notes
    -----
    skchange change detectors intentionally differ from sklearn's typical estimator
    assumptions in a few places:

    - Time-series estimators are order-sensitive by design.
    - Detectors require at least two samples.
    """
    tags = estimator.__sklearn_tags__()
    is_change_detector = tags.change_detector_tags is not None
    if not is_change_detector:
        return {}

    return {
        "check_methods_subset_invariance": (
            "Subset checks may create single-sample inputs, but change detection "
            "requires at least two samples."
        ),
        "check_methods_sample_order_invariance": (
            "Change detection is for time series; sample order is semantically "
            "meaningful and not invariant under permutation."
        ),
    }


def test_new_api_estimator_discovery_not_empty():
    """Sanity check that at least one estimator is discovered."""
    assert len(NEW_API_ESTIMATORS) > 0


@parametrize_with_checks(
    NEW_API_ESTIMATORS,
    expected_failed_checks=_expected_failed_checks,
)
def test_new_api_estimators_sklearn_compatibility(estimator, check):
    """Run sklearn's estimator checks on discovered ``new_api`` estimators."""
    check(estimator)
