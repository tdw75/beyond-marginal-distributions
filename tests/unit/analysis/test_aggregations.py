import pytest

from src.analysis.aggregations import aggregate_distributions


def test_aggregate_distributions():
    dists = {
        "adapter1": {"Q1": {"1": 0.5, "2": 0.5}, "Q2": {"1": 0.7, "2": 0.3}},
        "adapter2": {"Q1": {"1": 0.2, "2": 0.8}, "Q2": {"1": 0.4, "2": 0.6}},
    }
    expected = {
        "Q1": {"1": 0.35, "2": 0.65},
        "Q2": {"1": 0.55, "2": 0.45},
    }
    output = aggregate_distributions(dists)

    for qnum in expected:
        assert output[qnum] == pytest.approx(expected[qnum])
