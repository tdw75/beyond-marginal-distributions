import pandas as pd

from src.data.variables import QNum, ResponseMap


def mark_is_scale_flipped(responses: list[str]):
    """
    responses : list of generated responses for a single survey question
    """
    return [i % 2 == 1 for i in range(len(responses))]


def key_as_int(response_map: dict[QNum, ResponseMap]) -> dict:
    return {
        qnum: {int(k): v for k, v in resp.items()}
        for qnum, resp in response_map.items()
    }


def reformat_index(index: pd.Index | list) -> pd.Index:
    if not isinstance(index, pd.Index):
        index = pd.Index(index)
    return index.str.replace("_", " ").str.title()
