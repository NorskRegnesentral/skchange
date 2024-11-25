import numpy as np
import pytest

from skchange.base.base_interval_scorer import BaseIntervalScorer


class ConcreteIntervalEvaluator(BaseIntervalScorer):
    def _evaluate(self, cuts):
        return np.array([np.sum(self._X[cut[0] : cut[-1]]) for cut in cuts])


def test_fit():
    evaluator = ConcreteIntervalEvaluator()
    X = np.array([1, 2, 3, 4, 5])
    evaluator.fit(X)
    assert evaluator._is_fitted
    assert np.array_equal(evaluator._X, X)


def test_evaluate():
    evaluator = ConcreteIntervalEvaluator()
    X = np.array([1, 2, 3, 4, 5])
    evaluator.fit(X)
    cuts = np.array([[0, 2], [2, 5]])
    values = evaluator.evaluate(cuts)
    expected_values = np.array([3, 12])
    assert np.array_equal(values, expected_values)


def test_min_size():
    evaluator = ConcreteIntervalEvaluator()
    assert evaluator.min_size == 1


def test_check_cuts():
    evaluator = ConcreteIntervalEvaluator()
    cuts = np.array([[0, 2], [2, 5]])
    checked_cuts = evaluator._check_cuts(cuts)
    assert np.array_equal(checked_cuts, cuts)


def test_not_implemented_evaluate():
    evaluator = BaseIntervalScorer()
    with pytest.raises(NotImplementedError):
        evaluator._evaluate(np.array([[0, 2]]))
