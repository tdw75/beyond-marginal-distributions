import json
import os

import pandas as pd

from src.analysis.aggregations import DistDict
from src.analysis.visualisations import RENAME_MAP, reformat_index
from src.data.variables import QNum, ResponseMap, remap_response_maps
from src.utils import key_as_int


def create_subdirectory(directory: str, subdirectory: str) -> str:
    """
    Create a subdirectory if it does not exist.
    """
    full_path = os.path.join(directory, subdirectory)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path


def load_response_maps(directory: str = "../data_files") -> dict[QNum, ResponseMap]:
    with open(
        os.path.join(directory, "variables/response_map_original.json"), "r"
    ) as f1:
        response_map = key_as_int(json.load(f1))
        response_map = remap_response_maps(response_map)
        response_map = {k: v for k, v in response_map.items() if k != "Q215"}

    return response_map


def save_latex_table(df: pd.DataFrame, directory: str, name: str, **kwargs):
    df = df.rename(columns=RENAME_MAP, errors="ignore")
    df.index = reformat_index(df.index)
    df.to_latex(os.path.join(directory, name), **kwargs)


def save_response_distributions(dists: DistDict, data_directory: str, grouping: str):
    with open(
        os.path.join(data_directory, f"{grouping}-response-dists.json"), "w"
    ) as f:
        json.dump(dists, f)
