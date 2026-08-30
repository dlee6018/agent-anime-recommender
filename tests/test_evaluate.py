import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import evaluate  # noqa: E402

EVAL = {1: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        2: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]}


def test_perfect():
    res = evaluate(lambda q, k: EVAL[q[0]][:k], EVAL)
    assert res["precision_at_k"] == 1.0
    assert res["mrr"] == 1.0


def test_partial():
    # 2 of 5 in truth for query 1; 0 of 5 for query 2
    fn = lambda q, k: [10, 11, 99, 98, 97] if q == [1] else [91, 92, 93, 94, 95]
    res = evaluate(fn, EVAL)
    assert res["precision_at_k"] == pytest.approx((0.4 + 0.0) / 2)
    assert res["mrr"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2


def test_query_leak_raises():
    with pytest.raises(AssertionError):
        evaluate(lambda q, k: [q[0], 99, 98, 97, 96], EVAL)


def test_empty_raises():
    with pytest.raises(ValueError):
        evaluate(lambda q, k: [], {})
