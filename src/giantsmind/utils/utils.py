import os
from pathlib import Path
from typing import Any, Callable, List, Tuple


def set_env_vars():
    env_file = Path(__file__).parent.parent.parent.parent / ".env"

    if not env_file.exists():
        raise Exception(".env file not found.")

    for line in env_file.read_text().split("\n"):
        if not line:
            continue
        key, value = line.split("=")
        os.environ[key] = value


def get_exist_absent(
    list_: List[Any], func_exist: Callable[[List[Any]], bool]
) -> Tuple[List[Any], List[int], List[Any], List[int]]:
    exist_flags = func_exist(list_)
    exist = [(i, elt) for i, (elt, exist) in enumerate(zip(list_, exist_flags)) if exist]
    list_exist, index_exist = [], []
    if len(exist) != 0:
        index_exist, list_exist = zip(*exist)

    to_process = [(i, elt) for i, (elt, exist) in enumerate(zip(list_, exist_flags)) if not exist]
    list_to_process, index_to_process = [], []
    if len(to_process) != 0:
        index_to_process, list_to_process = zip(*to_process)

    return list_exist, index_exist, list_to_process, index_to_process


def reorder_merge_lists(
    docs1: List[Any], docs2: List[Any], index1: List[int], index2: List[int]
) -> List[Any]:
    docs_new = [None] * (len(docs1) + len(docs2))
    for i, doc in zip(index1, docs1):
        docs_new[i] = doc
    for i, doc in zip(index2, docs2):
        docs_new[i] = doc
    return docs_new
