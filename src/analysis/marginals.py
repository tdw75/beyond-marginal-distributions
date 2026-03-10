import json
import os
from collections import Counter
from typing import Callable

import pandas as pd

from src.analysis.aggregations import DataDict, steered_models, DistDict
from src.analysis.io import save_latex_table
from src.analysis.metrics import (
    calculate_dissimilarity,
    calculate_variance,
    prepare_distributions_single,
)
from src.analysis.responses import FrequencyDist
from src.analysis.visualisations import plot_distance_heatmap, RENAME_MAP
from src.data.variables import QNum, ResponseMap
from src.simulation.models import ModelName, AdapterName


def compare_marginal_response_dists(
    dists: DistDict, metric_directory: str, grouping: str
):
    dissimilarity = {
        n: get_metric(d, calculate_dissimilarity) for n, d in dists.items()
    }
    variances = {n: get_variance(d) for n, d in dists.items()}

    flatten_to_df_long(variances).to_csv(
        os.path.join(metric_directory, f"{grouping}-variances.csv")
    )
    flatten_to_df_long(dissimilarity).to_csv(
        os.path.join(metric_directory, f"{grouping}-dissimilarity.csv")
    )


def get_metric(
    dists: dict[ModelName, dict[QNum, FrequencyDist]], metric_fn
) -> dict[str, pd.Series]:
    return {
        "opinion_gpt": pd.Series(metric_fn(dists["opinion_gpt"], dists["true"])),
        "persona": pd.Series(metric_fn(dists["persona"], dists["true"])),
        "base": pd.Series(metric_fn(dists["base"], dists["true"])),
    }


def get_variance(
    dists: dict[ModelName, dict[QNum, FrequencyDist]],
) -> dict[str, pd.Series]:
    return {
        "opinion_gpt": pd.Series(calculate_variance(dists["opinion_gpt"])),
        "persona": pd.Series(calculate_variance(dists["persona"])),
        "true": pd.Series(calculate_variance(dists["true"])),
        "base": pd.Series(calculate_variance(dists["base"])),
    }


def flatten_to_df_long(metric_results: dict[str, dict]) -> pd.DataFrame:
    records = []
    for sg, metrics in metric_results.items():
        for model, vals in metrics.items():
            for qnum, value in pd.Series(vals).items():
                records.append(
                    {"number": qnum, "group": sg, "model": model, "value": value}
                )
    cols = ["number", "group", "model", "value"]
    df = pd.DataFrame(records)
    return df[cols].sort_values(cols[1:]).reset_index(drop=True)


def generate_modal_collapse_analysis(
    data_dict: DataDict,
    base: pd.DataFrame,
    metrics_directory: str,
    latex_directory: str,
):
    with open(os.path.join(metrics_directory, "degenerate-dists.json"), "w") as f1:
        deg, counts = find_degenerate_dists(data_dict, base)
        json.dump(deg, f1)

    with open(os.path.join(metrics_directory, "degenerate-qnums.json"), "w") as f2:
        qnums = find_degenerate_questions(deg)
        json.dump(qnums, f2)

    save_latex_table(
        counts.T,
        latex_directory,
        "degenerate-counts-table.tex",
        column_names=RENAME_MAP,
    )


def find_degenerate_dists(data: dict, base: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Identify degenerate distributions across all models."""

    def _find_single(df: pd.DataFrame) -> list[str]:
        return [
            qnum for qnum in df.columns if df.loc[df[qnum] >= 0, qnum].nunique() == 1
        ]

    degenerate_dists, counts = {"base": _find_single(base)}, {}
    for s, _ in data.items():
        degenerate_dists[s] = {m: _find_single(data[s][m]) for m in steered_models}
        counts[s] = {
            m: pd.Series(degenerate_dists[s][m]).count()
            for m in degenerate_dists[s].keys()
        }

    return degenerate_dists, pd.DataFrame(counts)


def find_degenerate_questions(
    degenerate_dists: dict[ModelName, dict[AdapterName, list[QNum]]],
) -> dict[QNum, int]:
    """
    Find questions that have degenerate distributions across all models.
    """
    degenerate_questions = Counter()
    for adapter, models in degenerate_dists.items():
        if adapter == "base":
            degenerate_questions.update(models)
        else:
            for qnums in models.values():
                degenerate_questions.update(qnums)

    return degenerate_questions


def get_response_distributions(
    dfs: dict[str, pd.DataFrame], response_map: dict[QNum, ResponseMap]
):
    return {
        "opinion_gpt": prepare_distributions_single(dfs["opinion_gpt"], response_map),
        "persona": prepare_distributions_single(dfs["persona"], response_map),
        "base": prepare_distributions_single(dfs["base"], response_map),
        "true": prepare_distributions_single(dfs["true"], response_map),
    }


def generate_cross_comparison(
    data_dict: DataDict,
    response_map: dict[QNum, ResponseMap],
    graph_directory: str,
    grouping: str,
):
    name, fn, cmap = ("Dissimilarity", calculate_dissimilarity, "Blues")
    cross = get_cross_distance(data_dict, fn, response_map)
    plot_distance_heatmap(
        cross, name, cmap=cmap, save_directory=graph_directory, grouping=grouping
    )


def get_cross_distance(
    data_dict: dict,
    metric_fn: Callable,
    response_map: dict[QNum, ResponseMap],
    data_name: str = "true",
) -> pd.DataFrame:
    cross = {}
    data_dict = data_dict.copy()
    # data_dict["Base Phi 3"] = {data_name: base}
    for s1, d1 in data_dict.items():
        cross[s1] = {}
        for s2, d2 in data_dict.items():
            cross[s1][s2] = pd.Series(
                metric_fn(d1[data_name], d2[data_name], response_map)
            ).mean()

    return pd.DataFrame(cross).T.round(4)


def generate_invalid_response_analysis(
    subgroup_data: DataDict, metrics_directory: str, latex_directory: str
):
    def _get_invalid_means(df: pd.DataFrame) -> pd.Series:
        return (df == -1).mean()

    def _get_invalid_by_qnum(df: pd.DataFrame) -> dict[QNum, float]:
        return _get_invalid_means(df)[lambda x: x > 0.1].to_dict()

    invalid_counts = {}
    invalid_totals = {}
    for sg, dd in subgroup_data.items():
        invalid_counts[sg] = {m: _get_invalid_by_qnum(dd[m]) for m in dd.keys()}
        invalid_totals[sg] = {m: _get_invalid_means(dd[m]).mean() for m in dd.keys()}

    invalid_totals = pd.DataFrame(invalid_totals).T
    save_latex_table(
        invalid_totals,
        latex_directory,
        "invalid-counts-total.tex",
        column_names=RENAME_MAP,
        float_format="%.3f",
    )
    invalid_totals.to_csv(os.path.join(metrics_directory, "invalid-counts-total.csv"))

    with open(
        os.path.join(metrics_directory, "invalid-counts-question.json"), "w"
    ) as f:
        json.dump(invalid_counts, f)

    common_qnums = {m: Counter() for m in steered_models + ["true", "base"]}
    for sg, models in invalid_counts.items():
        for m, qnums in models.items():
            common_qnums[m].update(qnums.keys())

    for m, counts in common_qnums.items():
        common_qnums[m] = {i: counts[i] for i in sorted(counts.keys())}

    with open(
        os.path.join(metrics_directory, "common-invalid-questions.json"), "w"
    ) as f:
        json.dump(common_qnums, f)
