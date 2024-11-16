import sqlite3
from sqlite3 import Cursor
from typing import Any, Callable, Dict, List, Tuple

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


def _process_results(results: List[Tuple[str]]) -> List[Dict[str, Any]]:
    res_list_dict = []
    for result in results:
        res_list_dict.append(
            {
                "title": result[0],
                "journal": result[1],
                "publication_date": result[2],
                "authors": result[3],
                "paper_id": result[4],
                "url": result[5],
            }
        )
    return res_list_dict


def execute_metadata_query(
    query: str,
    preprocess_query: Callable[[str], str],
    process_results: Callable[[List[Tuple]], List[str]],
    db_connector: DatabaseConnection,
    config: DatabaseConfig,
) -> List[Dict[str, Any]]:
    """Execute metadata query with improved testability."""
    processed_query = preprocess_query(query)

    with db_connector.connect(config.path) as connection:
        setup_database_connection(connection, config.db_functions)
        cursor = connection.cursor()

        initial_results = execute_query(cursor, processed_query)
        paper_ids = [tuple_[0] for tuple_ in initial_results]

        paper_ids_txt = create_paper_ids_clause(paper_ids)
        papers_query = get_papers_query(paper_ids_txt)
        papers_metadata = execute_query(cursor, papers_query)

    return process_results(papers_metadata)
