import pandas as pd
import pytest

from simulation.models import DimensionName
from src.analysis.aggregations import (
    _get_empirical_dimension_weights,
    get_survey_weights_for_dimension,
    DataDict,
)


def test_convert_empirical_weight_to_respondent_weight(mock_subgroup_data, weights_exp):
    weights = get_survey_weights_for_dimension(mock_subgroup_data)
    expected = {
        "leaning": {k: v for k, v in zip(weights_exp["leaning"].index, [0.5, 1.5])},
        "country": {
            k: v for k, v in zip(weights_exp["country"].index, [0.75, 1.25, 1.5, 0.5])
        },
        "sex": {k: v for k, v in zip(weights_exp["sex"].index, [0.8, 1.2])},
        "age": {k: v for k, v in zip(weights_exp["age"].index, [1.25, 0.75])},
    }
    assert weights == expected


def test_get_empirical_dimension_weights(mock_subgroup_data, weights_exp):

    weights = _get_empirical_dimension_weights(mock_subgroup_data)
    for dim in weights_exp:
        pd.testing.assert_series_equal(weights[dim], weights_exp[dim])


@pytest.fixture
def mock_subgroup_data() -> DataDict:
    return {
        "liberal": {"true": pd.DataFrame(index=range(100))},
        "conservative": {"true": pd.DataFrame(index=range(300))},
        "german": {"true": pd.DataFrame(index=range(150))},
        "american": {"true": pd.DataFrame(index=range(250))},
        "middle_east": {"true": pd.DataFrame(index=range(300))},
        "latin_america": {"true": pd.DataFrame(index=range(100))},
        "men": {"true": pd.DataFrame(index=range(400))},
        "women": {"true": pd.DataFrame(index=range(600))},
        "people_over_30": {"true": pd.DataFrame(index=range(500))},
        "old_people": {"true": pd.DataFrame(index=range(300))},
    }


@pytest.fixture
def weights_exp() -> dict[DimensionName, pd.Series]:
    return {
        "leaning": pd.Series(
            [1 / (n1 := 4), 3 / n1], index=["liberal", "conservative"]
        ),
        "country": pd.Series(
            [1.5 / (n2 := 8), 2.5 / n2, 3 / n2, 1 / n2],
            index=["german", "american", "middle_east", "latin_america"],
        ),
        "sex": pd.Series([4 / (n3 := 10), 6 / n3], index=["men", "women"]),
        "age": pd.Series(
            [5 / (n4 := 8), 3 / n4], index=["people_over_30", "old_people"]
        ),
    }
