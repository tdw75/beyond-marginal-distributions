import os

import pandas as pd

from src.analysis.metrics import prepare_distributions_single
from src.analysis.responses import (
    get_true_responses_for_subgroup,
    get_model_responses_for_subgroup,
    FrequencyDist,
)
from src.data.variables import QNum, ResponseMap
from src.demographics.base import BaseSubGroup
from src.demographics.config import categories, category_to_question, dimensions
from src.simulation.models import AdapterName, ModelName, DimensionName

steered_models = ["opinion_gpt", "persona"]
all_models = steered_models + ["base"]
DataDict = dict[
    AdapterName | DimensionName, dict[ModelName, pd.DataFrame]
]  # subgroup -> model -> DataFrame

DistDict = dict[AdapterName, dict[ModelName, dict[QNum, FrequencyDist]]]


def collate_subgroup_data(
    df_true: pd.DataFrame,
    df_sim: pd.DataFrame,
    df_base: pd.DataFrame,
    subgroup: type[BaseSubGroup] | list[type[BaseSubGroup]],
    qnums: list[QNum],
) -> dict[str, pd.DataFrame]:
    # qnums columns, obs rows

    return {
        "true": pd.DataFrame(get_true_responses_for_subgroup(df_true, subgroup, qnums)),
        "opinion_gpt": pd.DataFrame(
            get_model_responses_for_subgroup(df_sim[df_sim["is_lora"]], subgroup, qnums)
        ),
        "persona": pd.DataFrame(
            get_model_responses_for_subgroup(
                df_sim[~df_sim["is_lora"]], subgroup, qnums
            )
        ),
        "base": df_base,
    }


def aggregate_distributions(
    dists: dict[AdapterName, dict[QNum, FrequencyDist]],
    weights: pd.Series | None = None,
) -> dict[QNum, FrequencyDist]:
    """
    Aggregate response distributions across subgroups for each question,
    e.g. for each question, calculate the mean distribution across subgroups.
    returns a dict mapping each question to its aggregated distribution.
    """

    uniform_weights = pd.Series({adapter: 1 / len(dists) for adapter in dists})
    weights = weights if weights is not None else uniform_weights
    weights = weights.reindex(dists.keys())

    aggregated = {}
    for qnum in next(iter(dists.values())).keys():
        freq_dists = pd.DataFrame(
            {k: pd.Series(dists[k][qnum]) for k in weights.keys()}
        )

        aggregated[qnum] = (freq_dists * weights).sum(axis=1).to_dict()
    return aggregated


def aggregate_distributions_by_dimension(
    true_data: dict[DimensionName, pd.DataFrame],
    subgroup_response_dists: DistDict,
    response_maps: dict[QNum, ResponseMap],
) -> DistDict:
    """
    Aggregate subgroup response distributions by dimension,
    using the provided weights for each subgroup within each dimension.
    """
    # todo: scratch this approach, just aggregate data by dimension then get dists using a weights column
    # todo: add base
    dimension_dists = {
        d: {m: {} for m in steered_models + ["true"]} for d in dimensions
    }

    for dim, subgroups in dimensions.items():
        subgroups = [s.ADAPTER for s in subgroups]

        # aggregate true data for dimension first, get empirical weighting
        dim_true_df = pd.concat(true_data[s] for s in subgroups)
        dim_counts = pd.Series({s: true_data[s].shape[0] for s in subgroups})
        dim_weights = dim_counts / dim_counts.sum()  # normalise to sum to 1
        dimension_dists[dim]["true"] = prepare_distributions_single(
            dim_true_df, response_maps
        )

        # then aggregate model distributions for dimension using weights
        for model in steered_models:
            dists = {sg: subgroup_response_dists[sg][model] for sg in subgroups}
            dimension_dists[dim][model] = aggregate_distributions(dists, dim_weights)

    return dimension_dists


def aggregate_data_by_category(
    data_dict: DataDict, base: pd.DataFrame, true: pd.DataFrame
) -> dict:

    all_qnums = set(base.columns)
    cat_dict = {c: {m: [] for m in steered_models + ["true"]} for c in categories}

    for cat, qnums in category_to_question.items():
        qnums = all_qnums.intersection(qnums)
        for sg, sources in data_dict.items():
            for model, df in sources.items():
                if model == "base":
                    continue
                elif model == "true":
                    df = true
                df_loop = df.filter(items=qnums)
                df_loop.index = [f"{sg}_{i}" for i in df_loop.index]
                cat_dict[cat][model].append(df_loop)
            cat_dict[cat]["base"] = [base.filter(items=qnums)]

    cat_dict = {
        c: {m: pd.concat(dfs) for m, dfs in models.items()}
        for c, models in cat_dict.items()
    }
    return cat_dict


def persist_data_dict(data_dict: DataDict, directory: str, grouping: str):
    for sg, models in data_dict.items():
        for model, df in models.items():
            if model != "base":
                df.to_csv(
                    os.path.join(directory, f"{grouping}-{model}-{sg}-responses.csv")
                )
    data_dict[sg]["base"].to_csv(
        os.path.join(directory, f"{grouping}-base-responses.csv")
    )
