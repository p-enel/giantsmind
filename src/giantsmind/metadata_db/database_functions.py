from dataclasses import dataclass
from sqlite3 import Connection
from typing import Callable, List

from sqlalchemy import event

from giantsmind.metadata_db.models import DatabaseFunction
from giantsmind.metadata_db.string_utils import author_name_distance, levenshtein

# Define standard database functions
levenshtein_func = DatabaseFunction(
    "levenshtein",
    2,
    levenshtein,
)

author_name_distance_func = DatabaseFunction(
    "author_name_distance",
    2,
    author_name_distance,
)


def connect_function_sqlite(connection: Connection, custom_functions: List[DatabaseFunction]) -> None:
    for func in custom_functions:
        connection.create_function(func.name, func.num_params, func.func)


def setup_db_functions(engine):
    @event.listens_for(engine, "connect")
    def string_match(conn, rec):
        conn.create_function("levenshtein", 2, levenshtein)

    @event.listens_for(engine, "connect")
    def author_match(conn, rec):
        conn.create_function("author_name_distance", 2, author_name_distance)
