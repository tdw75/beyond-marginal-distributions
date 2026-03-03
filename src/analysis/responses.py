import pandas as pd

from src.data.variables import QNum, ResponseMap, ordinal_qnums
from src.demographics.base import BaseSubGroup

FrequencyDist = dict[int, int] | dict[int, float]
ResponseSupport = list[int]
Dimension = list[type[BaseSubGroup]]


def get_true_responses_for_subgroup(
    df: pd.DataFrame, subgroup: type[BaseSubGroup] | Dimension, qnums: list[QNum]
) -> dict[QNum, pd.Series]:
    df = df.copy()
    is_subgroup = _get_subgroup_filter_true(df, subgroup)
    responses = {
        qnum: df.loc[is_subgroup, qnum].reset_index(drop=True) for qnum in qnums
    }
    responses["weight"] = df.loc[is_subgroup, "W_WEIGHT"].reset_index(drop=True)
    return responses


def get_model_responses_for_subgroup(
    df: pd.DataFrame,
    subgroup: type[BaseSubGroup] | Dimension,
    qnums: list[QNum],
    response_col: str = "final_response",
) -> dict[str, pd.Series]:
    df = df.copy()
    is_subgroup = _get_subgroup_filter_model(df, subgroup)
    return {
        qnum: df.loc[is_subgroup & (df["number"] == qnum), response_col].values
        for qnum in qnums
    }


def get_base_model_responses(df_sim: pd.DataFrame, qnums: list[QNum]) -> pd.DataFrame:
    df = df_sim.loc[df_sim["number"].isin(qnums), ["number", "final_response"]].copy()
    df["idx"] = df.groupby("number").cumcount()

    out = df.pivot(
        index="idx",
        columns="number",
        values="final_response",
    )

    return out.reindex(columns=qnums)


def _get_subgroup_filter_true(
    df: pd.DataFrame, subgroup: type[BaseSubGroup] | list[type[BaseSubGroup]]
) -> pd.Series:
    if subgroup is None:
        is_subgroup = pd.Series([True] * len(df), index=df.index)
    elif isinstance(subgroup, list):
        is_subgroup = pd.Series([False] * len(df), index=df.index)
        for sg in subgroup:
            is_subgroup |= sg.filter_true(df)
    else:
        is_subgroup = subgroup.filter_true(df)
    return is_subgroup


def _get_subgroup_filter_model(
    df: pd.DataFrame, subgroup: type[BaseSubGroup] | list[type[BaseSubGroup]]
) -> pd.Series:
    if isinstance(subgroup, list):
        is_subgroup = pd.Series([False] * len(df), index=df.index)
        for sg in subgroup:
            is_subgroup |= sg.filter_model(df)
    else:
        is_subgroup = subgroup.filter_model(df)
    return is_subgroup


def get_response_distribution(
    responses: pd.DataFrame,
    response_maps: dict[QNum, ResponseMap],
    is_normalize: bool = True,
    is_include_invalid: bool = False,
) -> dict[QNum, FrequencyDist]:

    dists = {}
    for qnum in set(responses.columns).intersection(response_maps.keys()):
        observations = responses[qnum]
        support = list(response_maps[qnum].keys())

        if not is_include_invalid:
            support = [x for x in support if x > -1]
            observations = observations[observations > -1]

        counts = observations.value_counts(normalize=is_normalize, sort=False)
        counts = counts.reindex(support, fill_value=0)
        dists[qnum] = {
            int(k): float(v) if is_normalize else int(v) for k, v in counts.items()
        }

    return dists


def get_response_distribution_weighted(
    responses: pd.DataFrame,
    response_maps: dict[QNum, ResponseMap],
    is_normalize: bool = True,
    is_include_invalid: bool = False,
) -> dict[QNum, FrequencyDist]:

    # todo: add weights for dimension aggregation/distributions
    dists = {}
    weights = responses["weight"]
    for qnum in set(responses.columns).intersection(response_maps.keys()) - {"weight"}:
        observations = responses[qnum]
        support = list(response_maps[qnum].keys())

        if not is_include_invalid:
            support = [x for x in support if x > -1]
            is_valid = observations > -1
            observations = observations[is_valid]
            weights_masked = weights[is_valid]
        else:
            weights_masked = weights

        counts = get_weighted_value_counts(observations, weights_masked, is_normalize)
        counts = counts.reindex(support, fill_value=0)
        dists[qnum] = {
            int(k): float(v) if is_normalize else int(v) for k, v in counts.items()
        }

    return dists


def get_weighted_value_counts(
    observations: pd.Series, weights: pd.Series, is_normalize: bool
) -> pd.Series:
    counts = (
        pd.DataFrame({"resp": observations, "w": weights}).groupby("resp")["w"].sum()
    )
    if is_normalize:
        counts = counts / counts.sum()
    return counts


def remove_weight_col(qnums: list[str]) -> list[str]:
    return [q for q in qnums if q != "weight"]


def get_support_minimum(
    response_maps: dict[QNum, ResponseMap], is_just_ordinal: bool = True
) -> pd.Series:
    qnums = ordinal_qnums() if is_just_ordinal else response_maps.keys()
    mins = {}
    for qnum, response_map in response_maps.items():
        if qnum in qnums:
            support = [
                k for k in response_map.keys() if k >= 0
            ]  # only consider valid responses
            mins[qnum] = min(support)

    return sort_by_qnum_index(pd.Series(mins))


def get_support_diameter(
    response_maps: dict[QNum, ResponseMap], is_just_ordinal: bool = True
) -> pd.Series:
    qnums = ordinal_qnums() if is_just_ordinal else response_maps.keys()
    diameters = {}
    for qnum, response_map in response_maps.items():
        if qnum in qnums:
            support = sorted(
                [k for k in response_map.keys() if k >= 0]
            )  # only consider valid responses
            diameters[qnum] = max(support) - min(support)

    return sort_by_qnum_index(pd.Series(diameters))


def sort_by_qnum_index(
    df: pd.DataFrame | pd.Series, qnums: list[QNum] = None
) -> pd.DataFrame | pd.Series:
    if qnums is not None:
        df = df[qnums].copy()
    return df.sort_index(
        key=lambda x: x.str.extract(r"(\d+)", expand=False).astype(int)
    )
