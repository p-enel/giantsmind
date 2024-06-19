from pathlib import Path
from typing import Dict, List

from giantsmind.utils import local

ALLOWED_ID_TYPES = ["doi", "arxiv"]


Collections = Dict[str, List[str]]


def check_id(id: str, allowed_types: List[str] = ALLOWED_ID_TYPES) -> None:
    if ":" not in id or len(id.split(":")) != 2:
        raise ValueError("ID must be in the format 'type:id'")
    if id.split(":")[0] not in ["doi", "arxiv", "pmid"]:
        raise ValueError(f"ID type must be one of {allowed_types}")


def create_collection(id_list: List[str], name: str) -> Collections:
    for id in id_list:
        check_id(id)
    collections_path = Path(local.get_local_data_path()) / "collections.json"
    collections = local.load_collections(str(collections_path))
    collections[name] = id_list
    local.save_collections(collections, str(collections_path))
    return collections


def remove_collection(name: str) -> Collections:
    collections_path = Path(local.get_local_data_path()) / "collections.json"
    collections = local.load_collections(str(collections_path))
    if name not in collections:
        raise ValueError(f"Collection '{name}' not found.")
    del collections[name]
    local.save_collections(collections, str(collections_path))
    return collections
