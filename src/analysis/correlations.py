import os
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error as rmse

from src.analysis.aggregations import DataDict, steered_models
from src.analysis.io import save_latex_table
from src.analysis.responses import sort_by_qnum_index
from src.data.variables import non_ordinal_qnums
from src.demographics.config import category_to_question, question_to_category


def compare_correlation_structures(
    subgroup_data: DataDict,
    diameters: pd.Series,
    minimums: pd.Series,
    filename: str,
    root_directory: str = "../data_files",
) -> dict[str, dict[str, float]]:
    """
    Compare correlation structures between steered models (persona prompting and OpinionGPT) and true responses.
    This results in a comparison of question-level and category-level correlation matrices.
    """

    metrics_directory = os.path.join(root_directory, "results", filename, "metrics")

    question_means: dict[str, pd.DataFrame] = {
        mod: get_question_means(subgroup_data, mod, diameters, minimums)
        for mod in steered_models + ["true"]
    }
    category_means: dict[str, pd.DataFrame] = {
        mod: get_category_means(df) for mod, df in question_means.items()
    }
    means = {"question": question_means, "category": category_means}
    metrics = {}
    for name, data in means.items():
        corr_matrices = {
            m: construct_correlation_matrix(df.T) for m, df in data.items()
        }
        for m in data.keys():
            data[m].round(2).to_csv(
                os.path.join(metrics_directory, f"{name}-means-subgroup-{m}.csv")
            )
            corr_matrices[m].to_csv(
                os.path.join(
                    metrics_directory, f"{name}-correlation-matrix-subgroup-{m}.csv"
                )
            )

        true = np.array(corr_matrices["true"])
        metrics[name] = {}
        for model in steered_models:
            metrics[name][model] = calculate_correlation_metrics(
                true, np.array(corr_matrices[model])
            )
    return metrics


def compare_correlation_structures_bootstrap(
    subgroup_data: DataDict,
    diameters: pd.Series,
    minimums: pd.Series,
    filename: str,
    root_directory: str = "../data_files",
    n_bootstrap: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Compare correlation structures between steered models (persona prompting and OpinionGPT) and true responses, using bootstrap resampling to estimate confidence intervals.
    This results in a comparison of question-level and category-level correlation matrices.
    """
    start = time.time()
    question_metrics = {"opinion_gpt": [], "persona": []}
    category_metrics = {"opinion_gpt": [], "persona": []}
    for i in range(n_bootstrap):
        bootstrapped_data = bootstrap_subgroup_data(subgroup_data)
        if (i + 1) % 10 == 0:
            print(
                f"Bootstrap iteration {i+1}/{n_bootstrap} finished, {time.time() - start:.1f} seconds"
            )
        metrics = compare_correlation_structures(
            bootstrapped_data, diameters, minimums, filename, root_directory
        )
        for model in steered_models:
            question_metrics[model].append(pd.Series(metrics["question"][model]))
            category_metrics[model].append(pd.Series(metrics["category"][model]))

    return {
        "question": get_confidence_intervals(question_metrics),
        "category": get_confidence_intervals(category_metrics),
    }


def get_confidence_intervals(metrics: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:

    def _get_quantiles(df: pd.DataFrame) -> list:
        return [list(zip(df.quantile(0.25).round(3), df.quantile(0.975).round(3)))]

    ci = []
    for mod in steered_models:
        concatenated = pd.concat(metrics[mod], axis=1).T
        ci.append(
            pd.DataFrame(
                _get_quantiles(concatenated), columns=concatenated.columns, index=[mod]
            )
        )
    return pd.concat(ci).T


def bootstrap_subgroup_data(subgroup_data: DataDict) -> DataDict:
    """Bootstrap resample the subgroup data by sampling with replacement within each subgroup."""
    bootstrapped_data = {}
    for sg, dd in subgroup_data.items():
        bootstrapped_data[sg] = {}
        for model, df in dd.items():
            if model == "base":
                continue
            elif model == "true":
                bootstrapped_data[sg][model] = df
            else:
                n = df.shape[0]  # should always be 500 but just in case, maybe assert
                bootstrapped_data[sg][model] = df.apply(
                    lambda col: col.iloc[
                        np.random.choice(n, size=n, replace=True)
                    ].values
                ).reset_index(drop=True)

    return bootstrapped_data


def save_correlation_metrics(
    corr_metrics: dict,
    grouping: str,
    filename: str,
    root_directory: str = "../data_files",
):
    corr_metrics = pd.DataFrame(corr_metrics)
    directory = os.path.join(root_directory, "results", filename, "latex")
    save_latex_table(
        corr_metrics,
        directory,
        f"{grouping}-correlation_metrics.tex",
        float_format="%.2f",
    )


def lower_bound(filename: str, root_directory: str = "../data_files") -> tuple:

    directory = os.path.join(root_directory, "results", filename, "metrics")

    def _shuffle_subgroups(df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(lambda x: x.sample(frac=1).values)

    lb_mean = {}
    lb_std = {}

    for grouping in ["question", "category"]:
        metrics = {}
        # also need category
        means = pd.read_csv(
            os.path.join(directory, f"{grouping}-means-subgroup-true.csv"),
            index_col=0,
        )
        corr_wvs = np.array(construct_correlation_matrix(means))

        for i in range(1000):
            means_shuffle = _shuffle_subgroups(means)
            corr_shuffle = construct_correlation_matrix(means_shuffle)
            metrics[i] = calculate_correlation_metrics(corr_wvs, np.array(corr_shuffle))

        metrics_df = pd.DataFrame(metrics).T

        lb_mean[grouping] = metrics_df.mean().round(3).to_dict()
        lb_std[grouping] = metrics_df.std().round(3).to_dict()

    return lb_mean, lb_std


def upper_bound(
    diameters: pd.Series, minimums: pd.Series, subgroup_data: DataDict
) -> tuple:

    ub_mean = {}
    ub_std = {}

    for grouping in ["question", "category"]:
        metrics = {}
        for i in range((B := 1000)):
            half_1, half_2 = split_true_data(subgroup_data)
            metrics[i] = split_half_analysis(
                half_1, half_2, diameters, minimums, grouping
            )
        metrics_df = pd.DataFrame(metrics).T
        ub_mean[grouping] = metrics_df.mean().round(3).to_dict()
        ub_std[grouping] = metrics_df.std().round(3).to_dict()

    return ub_mean, ub_std


@lru_cache(None)
def get_upper_triangle_from_dim(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return upper-triangle indices for an n x n matrix (k=1), cached by n."""
    return np.triu_indices(n, k=1)


def split_true_data(data: DataDict) -> tuple[DataDict, DataDict]:
    half_1 = {}
    half_2 = {}
    for sg, true in data.items():
        df = true["true"]
        n = len(df)
        if n % 2 != 0:
            n -= 1
        indices = np.random.permutation(n)
        mid = n // 2
        half_1[sg] = {"true": df.iloc[indices[:mid]]}
        half_2[sg] = {"true": df.iloc[indices[mid:]]}
    return half_1, half_2


def split_half_analysis(
    half_1: DataDict, half_2: DataDict, diameters, minimums, grouping: str
) -> dict[str, float]:

    means_1 = get_question_means(half_1, "true", diameters, minimums)
    means_2 = get_question_means(half_2, "true", diameters, minimums)
    if grouping == "category":
        means_1 = get_category_means(means_1)
        means_2 = get_category_means(means_2)

    corr_1 = np.array(construct_correlation_matrix(means_1))
    corr_2 = np.array(construct_correlation_matrix(means_2))

    return calculate_correlation_metrics(corr_1, corr_2)


def construct_correlation_matrix(means: pd.DataFrame) -> pd.DataFrame:
    """Construct correlation matrix from means DataFrame with rows as subgroups and columns as survey items."""
    return means.corr().round(3)


def calculate_correlation_metrics(
    true_corr: np.ndarray, model_corr: np.ndarray
) -> dict[str, float]:

    def _is_no_nans(m1: np.ndarray, m2: np.ndarray) -> bool:
        return ~np.isnan(m1).all(axis=1) & ~np.isnan(m2).all(axis=1)

    valid_idx = _is_no_nans(true_corr, model_corr)
    true = true_corr[np.ix_(valid_idx, valid_idx)]
    model = model_corr[np.ix_(valid_idx, valid_idx)]

    assert true.shape == model.shape, "Correlation matrices must have the same shape"
    assert _is_corr(true), "true_corr is not a valid correlation matrix"
    assert _is_corr(model), "model_corr is not a valid correlation matrix"

    iu = get_upper_triangle_from_dim(true.shape[1])
    x, y = true[iu], model[iu]

    corr_metrics = {}
    rho, _ = pearsonr(x, y)
    corr_metrics["pearson_r"] = rho.round(3)
    corr_metrics["rmse"] = np.round(rmse(x, y), 3)
    return corr_metrics


def _is_corr(matrix: np.ndarray) -> bool:
    """Check if a matrix is a valid correlation matrix (symmetric, ones on diagonal, square)."""
    if matrix.shape[0] != matrix.shape[1]:  # square
        return False
    if not np.allclose(matrix, matrix.T):  # symmetric
        return False
    if not np.allclose(np.diag(matrix), 1):  # ones on diagonal
        return False
    return True


def get_category_means(question_means: pd.DataFrame) -> pd.DataFrame:
    cat_means = {}
    qnums = list(question_means.index)
    categories = set(question_to_category[q] for q in qnums)

    for cat, qnums in category_to_question.items():
        if cat not in categories:
            continue
        df_loop = question_means.filter(items=qnums, axis=0)
        cat_means[cat] = df_loop.mean().round(2)

    return pd.DataFrame(cat_means).T


def get_question_means(
    subgroup_dict: DataDict,
    model_name: str,
    diameters: pd.Series,
    minimums: pd.Series,
    filter_val: str = None,
) -> pd.DataFrame:
    """
    Calculate the mean response for each question across all subgroups, normalised by the response diameter.
    """
    means = {}
    for sg, dd in subgroup_dict.items():
        df = dd[model_name].drop(
            columns=non_ordinal_qnums() + ["weight"], errors="ignore"
        )
        qnums = list(df.columns)
        if filter_val is not None:
            df = df.filter(like=filter_val)
        sg_means = df.replace(-1, np.nan).mean().round(2)  # drop invalid responses
        means[sg] = sort_by_qnum_index(sg_means) - sort_by_qnum_index(minimums, qnums)
        means[sg] = means[sg] / sort_by_qnum_index(diameters, qnums)

    return pd.DataFrame(means)
