import pandas as pd
import pytest

from src.analysis.aggregations import aggregate_distributions


@pytest.mark.parametrize(
    "weights, expected",
    [
        (None, {"Q1": {1: 0.35, 2: 0.65}, "Q2": {1: 0.55, 2: 0.45}}),
        (
            pd.Series({"adapter1": 0.75, "adapter2": 0.25}),
            {"Q1": {1: 0.425, 2: 0.575}, "Q2": {1: 0.625, 2: 0.375}},
        ),
    ],
)
def test_aggregate_distributions(weights, expected):
    dists = {
        "adapter1": {"Q1": {1: 0.5, 2: 0.5}, "Q2": {1: 0.7, 2: 0.3}},
        "adapter2": {"Q1": {1: 0.2, 2: 0.8}, "Q2": {1: 0.4, 2: 0.6}},
    }
    output = aggregate_distributions(dists, weights)

    for qnum in expected:
        assert output[qnum] == pytest.approx(expected[qnum])
