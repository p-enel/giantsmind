import sqlite3
from sqlite3 import Cursor
from typing import List, Tuple

from giantsmind.database.db_connection import (
    create_paper_ids_clause,
    get_papers_query,
    setup_database_connection,
)
from giantsmind.database.models import DatabaseConfig, DatabaseConnection
from giantsmind.utils.logging import logger


def execute_query(cursor: Cursor, query: str) -> List[Tuple]:
    """Execute SQL query and return results."""
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(f"Database error executing query: {query}\nError: {e}")
        raise


def execute_metadata_query(
    query: str,
    db_connector: DatabaseConnection,
    config: DatabaseConfig,
) -> List[Tuple]:
    """Execute metadata query with improved testability."""

    with db_connector.connect(config.path) as connection:
        setup_database_connection(connection, config.db_functions)
        cursor = connection.cursor()

        initial_results = execute_query(cursor, query)
        paper_ids = [tuple_[0] for tuple_ in initial_results]

        paper_ids_txt = create_paper_ids_clause(paper_ids)
        papers_query = get_papers_query(paper_ids_txt)
        papers_metadata = execute_query(cursor, papers_query)

    return papers_metadata
