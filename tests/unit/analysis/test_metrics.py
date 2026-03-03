import pytest
import pandas as pd
import numpy as np

from src.analysis.metrics import calculate_wasserstein, calculate_jensen_shannon


@pytest.mark.skip(reason="JS distance no longer used in project")
def test_js_identical(all_responses):
    model, _ = all_responses
    true, _ = all_responses
    result = calculate_jensen_shannon(model, true)
    assert result["Q1"] == 0.0
    assert result["Q2"] == 0.0


@pytest.mark.skip(reason="JS distance no longer used in project")
def test_js_different(all_responses):
    model, true = all_responses
    result = calculate_jensen_shannon(model, true)
    assert result["Q1"] == 0.0
    assert 0 < result["Q2"] < 1


@pytest.mark.skip(reason="JS distance no longer used in project")
def test_jensen_shannon_symmetry(all_responses):

    model, true = all_responses
    result1 = calculate_jensen_shannon(model, true)
    result2 = calculate_jensen_shannon(true, model)
    assert result1["Q1"] == result2["Q1"]
    assert result1["Q2"] == result2["Q2"]


def test_wasserstein_identical(all_responses):
    model, _ = all_responses
    true, _ = all_responses
    result = calculate_wasserstein(model, true)
    assert result["Q1"] == 0.0
    assert result["Q2"] == 0.0


def test_wasserstein_different(all_responses, response_maps):
    model, true = all_responses
    result = calculate_wasserstein(model, true)
    assert result["Q1"] == 0.0
    assert 0 < result["Q2"] < 1


def test_wasserstein_normalisation(extreme_responses, response_maps):
    model, true = extreme_responses
    result = calculate_wasserstein(model, true)
    assert np.isclose(result["Q1"], 1)


@pytest.fixture
def response_maps():
    return {
        "Q1": {-1: "M", 1: "A", 2: "B", 3: "C", 4: "D"},
        "Q2": {-1: "M", 1: "X", 2: "Y", 3: "Z", 4: "W"},
    }


@pytest.fixture
def all_responses():
    n1 = 6
    n2 = 5
    model = {
        "Q1": {key: freq / n1 for key, freq in zip([1, 2], [2, 4])},
        "Q2": {key: freq / n2 for key, freq in zip([1, 2, 3, 4], [1, 0, 3, 1])},
    }
    true = {
        "Q1": {
            key: freq / n1 for key, freq in zip([1, 2], [2, 4])
        },  # identical to model
        "Q2": {
            key: freq / n2 for key, freq in zip([1, 2, 3, 4], [3, 0, 0, 2])
        },  # different from model
    }
    return model, true


@pytest.fixture
def extreme_responses():
    # For normalisation test: all mass at min vs all at max
    model = {"Q1": {key: freq / 10 for key, freq in zip([1, 2, 3, 4], [10, 0, 0, 0])}}
    true = {"Q1": {key: freq / 10 for key, freq in zip([1, 2, 3, 4], [0, 0, 0, 10])}}
    return model, true
    # return pd.Series(model), pd.Series(true) # also works with DataFrame/Series instead of dict
