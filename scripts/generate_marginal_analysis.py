import json
import os
import sys
import time

import fire
import pandas as pd

print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())

from src.analysis.aggregations import (
    collate_subgroup_data,
    aggregate_data_by_category,
    DataDict,
    persist_data_dict,
    aggregate_data_by_dimension,
)
from src.analysis.io import create_subdirectory, save_response_distributions
from src.analysis.marginals import (
    generate_modal_collapse_analysis,
    generate_invalid_response_analysis,
    compare_marginal_response_dists,
    get_response_distributions,
)
from src.analysis.responses import get_base_model_responses
from src.data.variables import remap_response_maps
from src.demographics.config import subgroups
from src.simulation.experiment import load_experiment
from src.utils import key_as_int

# todo: currently does two jobs: collates/aggregates and runs marginal dist analysis - split?


def main(experiment_name: str, root_directory: str = ""):
    experiment = load_experiment(experiment_name, root_directory)

    start = time.time()

    simulation_directory = os.path.join(
        experiment.files["directory"], "results", experiment_name
    )
    sim = pd.read_csv(
        os.path.join(simulation_directory, f"{experiment_name}-clean.csv"), index_col=0
    )
    if "final_response" not in sim.columns:
        sim["final_response"] = sim["response_key"]
    sim = sim.loc[sim["number"] != "Q215"]  # not asked in USA
    all_qnums = list(sim["number"].unique())
    true = pd.read_csv(
        os.path.join(
            experiment.files["directory"], "WV7/WVS_Cross-National_Wave_7_csv_v6_0.csv"
        ),
        index_col=0,
    )

    with open(
        os.path.join(
            experiment.files["directory"], "variables/response_map_original.json"
        ),
        "r",
    ) as f1:
        response_map = key_as_int(json.load(f1))
        response_map = remap_response_maps(response_map)
        response_map = {k: v for k, v in response_map.items() if k != "Q215"}

    sim["subgroup"].fillna("none", inplace=True)
    base = get_base_model_responses(sim[sim["subgroup"] == "none"], all_qnums)
    print(f"Loaded data, {time.time() - start:.1f} seconds")
    subgroup_data: DataDict = {
        n: collate_subgroup_data(true, sim, base, s, all_qnums)
        for n, s in subgroups.items()
    }
    print(f"Aggregated subgroup data, {time.time() - start:.1f} seconds")
    dimension_data = aggregate_data_by_dimension(subgroup_data, base)
    print(f"Aggregated dimension data, {time.time() - start:.1f} seconds")
    category_data = aggregate_data_by_category(subgroup_data, base, true)
    print(f"Aggregated category data, {time.time() - start:.1f} seconds")
    metrics_directory = create_subdirectory(simulation_directory, "metrics")
    data_directory = create_subdirectory(simulation_directory, "data")
    latex_directory = create_subdirectory(simulation_directory, "latex")

    persist_data_dict(subgroup_data, data_directory, "subgroup")

    generate_modal_collapse_analysis(
        subgroup_data, base, metrics_directory, latex_directory
    )
    print(f"Finished modal collapse analysis, {time.time() - start:.1f} seconds")
    generate_invalid_response_analysis(
        subgroup_data, metrics_directory, latex_directory
    )
    print(f"Finished invalid response analysis, {time.time() - start:.1f} seconds")
    generate_invalid_response_analysis(
        category_data, metrics_directory, latex_directory
    )

    data_dict_map = {
        "subgroup": subgroup_data,
        "dimension": dimension_data,
        "category": category_data,
    }

    for grouping, data_dict in data_dict_map.items():
        dists = {
            n: get_response_distributions(d, response_map) for n, d in data_dict.items()
        }
        save_response_distributions(
            dists, create_subdirectory(simulation_directory, "data"), grouping
        )
        compare_marginal_response_dists(dists, metrics_directory, grouping)
        print(
            f"Finished model comparison metrics for {grouping}, {time.time() - start:.1f} seconds"
        )


if __name__ == "__main__":
    fire.Fire(main)
