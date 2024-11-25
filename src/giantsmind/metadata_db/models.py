from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Connection
from typing import Any, Callable, List, Protocol

from typing_extensions import Self


class DatabaseConnection(Protocol):
    """Database connection protocol with context manager support."""

    def connect(self) -> Connection: ...

    def close(self) -> None: ...


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


@dataclass
class DatabaseConfig:
    path: Path
    db_functions: List[DatabaseFunction]
