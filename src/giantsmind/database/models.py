import sqlite3
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Callable, List, Protocol


@dataclass
class DatabaseConfig:
    path: str
    db_functions: List[Callable]


class DatabaseConnection(Protocol):
    def connect(self, database: str) -> Connection: ...


class SQLiteConnection:
    """Basic SQLite database connection."""

    def connect(self, database: str) -> Connection:
        return sqlite3.connect(database)
