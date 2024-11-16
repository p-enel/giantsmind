from pathlib import Path
from giantsmind.utils import local
from dataclasses import dataclass
from typing import Any, Callable

local_data_path = Path(local.get_local_data_path())
local_data_path.mkdir(exist_ok=True)
DATABASE_PATH = local_data_path / "papers.db"

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"


@dataclass
class DatabaseFunction:
    """
    Represents a custom database function configuration.

    Attributes:
        name: Name of the function as it will be used in SQL
        num_params: Number of parameters the function accepts
        func: The Python function to be registered
    """

    name: str
    num_params: int
    func: Callable[..., Any]
