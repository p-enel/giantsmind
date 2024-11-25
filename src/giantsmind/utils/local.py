import json
import logging
import platform
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from platformdirs import user_data_dir, user_documents_dir

Collections = Dict[str, List[str]]


# def _get_potential_paths() -> Tuple[Path, Path]:
#     """Returns both primary and fallback paths"""
#     primary = Path(user_data_dir("giantsmind", appauthor=False))
#     if platform.system() == "Windows":
#         fallback = Path(user_documents_dir()) / "giantsmind"
#     else:
#         fallback = Path.home() / ".giantsmind"
#     return primary, fallback


# @lru_cache(maxsize=1)
# def get_local_data_path(get_potential_paths: Tuple[Path, Path] = _get_potential_paths) -> Path:
#     primary, fallback = _get_potential_paths()

#     # Check if data exists in fallback location
#     if fallback.exists() and any(fallback.iterdir()):
#         try:
#             # Try to migrate data to primary location if possible
#             primary.mkdir(exist_ok=True, parents=True)
#             for item in fallback.iterdir():
#                 if not (primary / item.name).exists():
#                     if item.is_file():
#                         shutil.copy2(item, primary / item.name)
#                     else:
#                         shutil.copytree(item, primary / item.name)
#             return primary
#         except PermissionError:
#             logging.warning(f"Cannot access {primary}. Using fallback at {fallback}")
#             return fallback

#     # Try primary location
#     try:
#         primary.mkdir(exist_ok=True, parents=True)
#         return primary
#     except PermissionError:
#         logging.warning(f"Cannot access {primary}. Using fallback at {fallback}")
#         fallback.mkdir(exist_ok=True, parents=True)
#         return fallback


def get_local_data_path() -> str:
    local_path = Path.home() / ".local" / "share" / "giantsmind"
    local_path.mkdir(exist_ok=True)
    return str(local_path)


def load_collections(path: str) -> Collections:
    if not Path(path).exists():
        return {}
    with open(path) as f:
        collections = json.load(f)
    return collections


def save_collections(collections: Collections, path: str) -> None:
    with open(path, "w") as f:
        json.dump(collections, f)


def load_metadata_df() -> pd.DataFrame:
    metadata_path = get_local_data_path() / "metadata.pkl"
    if metadata_path.exists():
        metadata_df = pd.read_pickle(metadata_path)
    else:
        metadata_df = pd.DataFrame()
    return metadata_df


def delete_metadata_df() -> None:
    metadata_path = get_local_data_path() / "metadata.pkl"
    if metadata_path.exists():
        metadata_path.unlink()


def save_metadata_df(metadata_df: pd.DataFrame) -> None:
    metadata_path = get_local_data_path() / "metadata.pkl"
    metadata_df.to_pickle(metadata_path)
