import os

import pandas as pd

from src.analysis.responses import (
    get_true_responses_for_subgroup,
    get_model_responses_for_subgroup,
    FrequencyDist,
)
from src.data.variables import QNum
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


def aggregate_data_by_dimension(subgroup_data: DataDict, base: pd.DataFrame) -> dict:
    """
    Aggregate response data across subgroups for each dimension,
    using the provided weights for each subgroup within each dimension.
    returns a dict mapping each dimension to its aggregated response data for each model.
    """
    weights = get_survey_weights_for_dimension(subgroup_data)
    dimension_data: DataDict = {
        d: {m: pd.DataFrame() for m in steered_models + ["true"]} for d in dimensions
    }
    for dim, subgroups in dimensions.items():
        for m in ["true"] + steered_models:

            names = [s.ADAPTER for s in subgroups]
            subgroup_dfs = [subgroup_data[s][m] for s in names]
            if m != "true":
                subgroup_dfs = [
                    df.assign(weight=weights[dim][s])
                    for s, df in zip(names, subgroup_dfs)
                ]

            dimension_data[dim][m] = pd.concat(subgroup_dfs).reset_index(drop=True)
        dimension_data[dim]["base"] = base.copy()
    return dimension_data


def _add_weight_column(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df["weight"] = weight
    return df


def aggregate_data_by_category(
    data_dict: DataDict, base: pd.DataFrame, true: pd.DataFrame
) -> dict:

    all_qnums = set(base.columns)
    cat_dict = {c: {m: [] for m in steered_models + ["true"]} for c in categories}

    for cat, qnums in category_to_question.items():
        cat_qnums = all_qnums.intersection(qnums)
        for sg, sources in data_dict.items():
            for model, df in sources.items():
                if model == "base":
                    continue
                elif model == "true":
                    df = true
                df_loop = df.filter(items=cat_qnums)
                df_loop.index = [f"{sg}_{i}" for i in df_loop.index]
                cat_dict[cat][model].append(df_loop)
            cat_dict[cat]["base"] = [base.filter(items=cat_qnums)]

    cat_dict = {
        c: {m: pd.concat(dfs) for m, dfs in models.items()}
        for c, models in cat_dict.items()
    }
    return cat_dict


def get_survey_weights_for_dimension(
    subgroup_data: DataDict,
) -> dict[DimensionName, dict[str, float]]:
    converted_weights = {}

    dimension_weights = _get_empirical_dimension_weights(subgroup_data)
    for dim, weights in dimension_weights.items():
        converted_weights[dim] = (weights * weights.shape[0]).to_dict()

    return converted_weights


def _get_empirical_dimension_weights(
    subgroup_data: DataDict,
) -> dict[DimensionName, pd.Series]:
    """
    Get weights for each dimension based on the empirical distribution of subgroups in the data.
    For example, if the "age" dimension has 3 subgroups (18-29, 30-44, 45+), and the data has 50% 18-29, 30% 30-44, and 20% 45+, then the weights for the "age" dimension would be [0.5, 0.3, 0.2].
    """
    dimension_weights = {}
    for dim_name, dim_subgroups in dimensions.items():
        dim_sg_names = [sg.ADAPTER for sg in dim_subgroups]
        subgroup_counts = pd.Series(0, index=dim_sg_names)
        for sg in dim_sg_names:
            subgroup_counts[sg] = subgroup_data[sg]["true"].shape[0]
        total = subgroup_counts.sum()
        dimension_weights[dim_name] = subgroup_counts / total
    return dimension_weights


def persist_data_dict(data_dict: DataDict, directory: str, grouping: str):
    # todo: move to io
    for sg, models in data_dict.items():
        for model, df in models.items():
            if model != "base":
                df.to_csv(
                    os.path.join(directory, f"{grouping}-{model}-{sg}-responses.csv")
                )
    data_dict[sg]["base"].to_csv(
        os.path.join(directory, f"{grouping}-base-responses.csv")
    )
