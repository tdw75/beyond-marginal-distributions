import pytest

from src.simulation.experiment import load_experiment, Experiment


@pytest.mark.parametrize(
    "experiment_name, expected",
    [
        (
            "unit_test",
            {
                "setup": {"name": "unit_test", "random_seed": 43},
                "files": {
                    "directory": "test_data_files",
                    "variables": "sample_variables.csv",
                    "subset": "unit_test_subset.json",
                },
                "simulation": {
                    "sample_size": 11,
                    "batch_size": 11,
                    "decoding_style": "unconstrained",
                    "base_model_name": "claude",
                    "temperature": 0.9,
                },
            },
        ),
        (
            "blank",
            {
                "setup": {"name": "blank", "random_seed": 43},
                "files": {
                    "directory": "test_data_files",
                    "variables": "sample_variables.csv",
                    "subset": "final_subset.json",
                },
                "simulation": {
                    "sample_size": 2000,
                    "batch_size": 32,
                    "decoding_style": "unconstrained",
                    "base_model_name": "claude",
                    "temperature": 0.9,
                },
            },
        ),
    ],
)
def test_load_experiment(experiment_name, expected):
    experiment = load_experiment(experiment_name, "test_data_files")
    assert experiment == Experiment(**expected)
