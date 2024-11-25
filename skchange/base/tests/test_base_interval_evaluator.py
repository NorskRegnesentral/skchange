import numpy as np
import pytest

from skchange.base.base_interval_scorer import BaseIntervalScorer


class ConcreteIntervalEvaluator(BaseIntervalScorer):
    def _evaluate(self, intervals):
        return np.array(
            [np.sum(self._X[interval[0] : interval[-1]]) for interval in intervals]
        )


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
    intervals = np.array([[0, 2], [2, 5]])
    values = evaluator.evaluate(intervals)
    expected_values = np.array([3, 12])
    assert np.array_equal(values, expected_values)


def test_min_size():
    evaluator = ConcreteIntervalEvaluator()
    assert evaluator.min_size == 1


def test_check_intervals():
    evaluator = ConcreteIntervalEvaluator()
    intervals = np.array([[0, 2], [2, 5]])
    checked_intervals = evaluator._check_intervals(intervals)
    assert np.array_equal(checked_intervals, intervals)


def test_not_implemented_evaluate():
    evaluator = BaseIntervalScorer()
    with pytest.raises(NotImplementedError):
        evaluator._evaluate(np.array([[0, 2]]))
