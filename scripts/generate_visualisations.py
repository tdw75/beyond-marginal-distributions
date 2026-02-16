import os
import sys

import fire
import pandas as pd

print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())


from src.analysis.io import create_subdirectory
from src.analysis.visualisations import (
    plot_model_metric_comparison,
    plot_model_metric_comparison_stacked,
)
from src.demographics.config import subgroups, dimensions, categories
from src.simulation.experiment import load_experiment

models = ["opinion_gpt", "persona", "base"]


def main(experiment_name: str, root_directory: str = ""):
    experiment = load_experiment(experiment_name, root_directory)

    simulation_directory = os.path.join(
        experiment.files["directory"], "results", experiment_name
    )
    simulation_directory = os.path.join(root_directory, simulation_directory)
    read_directory = create_subdirectory(simulation_directory, "metrics")
    save_directory = create_subdirectory(simulation_directory, "graphs")

    for metric in ["variances", "dissimilarity"]:
        sg_means, sg_05, sg_95 = load_and_calculate_metric_means(
            os.path.join(read_directory, f"subgroup-{metric}.csv"), "subgroup"
        )
        dim_means, dim_05, dim_95 = load_and_calculate_metric_means(
            os.path.join(read_directory, f"dimension-{metric}.csv"), "dimension"
        )
        plot_model_metric_comparison_stacked(
            sg_means,
            dim_means,
            (sg_05, sg_95),
            (dim_05, dim_95),
            save_directory=save_directory,
            subplot_scale=0.4,
            **METRIC_CONFIG[metric],
        )

        cat_means, cat_05, cat_95 = load_and_calculate_metric_means(
            os.path.join(read_directory, f"category-{metric}.csv"), "category"
        )
        plot_model_metric_comparison(
            cat_means.dropna(axis=0),
            (cat_05.dropna(axis=0), cat_95.dropna(axis=0)),
            save_directory=save_directory,
            **METRIC_CONFIG[metric],
            **GROUPING_CONFIG["category"],
        )


def load_and_calculate_metric_means(
    csv_path: str, grouping: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads a CSV and computes the mean for each column.
    """

    def q10(x):
        return x.quantile(0.1)

    def q90(x):
        return x.quantile(0.9)

    df = pd.read_csv(csv_path, index_col=0)
    grouped = df.groupby(["model", "group"], as_index=False).agg(
        {"value": ["mean", q10, q90]}
    )
    means_pivot = (
        grouped.pivot(index="group", columns="model", values=("value", "mean"))
        .round(4)
        .reindex(INDEX_ORDER[grouping])
    )

    q05_pivot = (
        grouped.pivot(index="group", columns="model", values=("value", "q10"))
        .round(4)
        .reindex(INDEX_ORDER[grouping])
    )

    q95_pivot = (
        grouped.pivot(index="group", columns="model", values=("value", "q90"))
        .round(4)
        .reindex(INDEX_ORDER[grouping])
    )

    cols = models + ["true"] if "true" in means_pivot.columns else models
    return means_pivot[cols], q05_pivot[cols], q95_pivot[cols]


INDEX_ORDER = {
    "subgroup": list(subgroups.keys()),
    "dimension": list(dimensions.keys()),
    "category": categories,
}


METRIC_CONFIG = {
    "variances": {"metric_name": "Response Variance", "xmax": 0.15},
    "dissimilarity": {"metric_name": "Dissimilarity", "xmax": 0.28},
}

GROUPING_CONFIG = {
    "subgroup": {"grouping": "subgroup", "subplot_scale": 0.4},
    "category": {"grouping": "category", "subplot_scale": 0.45},
    "dimension": {"grouping": "dimension", "subplot_scale": 0.55},
}


if __name__ == "__main__":
    fire.Fire(main)
