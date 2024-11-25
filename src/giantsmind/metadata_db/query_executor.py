import sqlite3
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Cursor
from typing import List, Tuple

from giantsmind.metadata_db.db_connection import DatabaseManager
from giantsmind.metadata_db.models import (
    DatabaseConfig,
    DatabaseConnection,
    DatabaseFunction,
)
from giantsmind.utils.logging import logger


@dataclass
class QueryValidationResult:
    is_valid: bool
    error_message: str | None = None


def execute_query(cursor: Cursor, query: str) -> List[Tuple]:
    """Execute SQL query and return results."""
    if not query.strip():
        raise ValueError("Empty query string")
    if not isinstance(cursor, Cursor):
        raise ValueError("Cursor must be an instance of sqlite3.Cursor")
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(f"Database error executing query: {query}\nError: {e}")
        raise


def create_paper_ids_clause(paper_ids: List[str]) -> str:
    """Create SQL clause for paper IDs filtering."""
    if len(paper_ids) == 1:
        return f"= '{paper_ids[0]}'"
    return f"IN {tuple(paper_ids)}"


def get_papers_query(paper_ids_txt: str) -> str:
    """Generate SQL query for fetching paper metadata."""
    return """SELECT
        papers.title,
        papers.journal_id,
        papers.publication_date,
        GROUP_CONCAT(authors.name, ', ') AS authors,
        papers.paper_id,
        papers.url
    FROM papers
    LEFT JOIN author_paper ON papers.paper_id = author_paper.paper_id
    LEFT JOIN authors ON author_paper.author_id = authors.author_id
    LEFT JOIN journals ON papers.journal_id = journals.journal_id
    WHERE papers.paper_id {}
    GROUP BY papers.paper_id, journals.name;
    """.format(
        paper_ids_txt
    )


class QueryExecutor:
    def __init__(self, connection_cls: DatabaseConnection, config: DatabaseConfig):
        self._validate_config(connection_cls, config)
        self.db_manager = DatabaseManager.get_instance(connection_cls, config)

    def _validate_config(self, connection_cls: DatabaseConnection, config: DatabaseConfig) -> None:
        if not isinstance(connection_cls, type):
            raise ValueError("connection_cls must be a class")
        if not isinstance(config, DatabaseConfig):
            raise ValueError("Invalid database configuration")
        if not config.path or not isinstance(config.path, Path):
            raise ValueError("Invalid database path")
        if not isinstance(config.db_functions, list) or not all(
            isinstance(func, DatabaseFunction) for func in config.db_functions
        ):
            raise ValueError("Invalid database functions")

    def _validate_query(self, query: str) -> QueryValidationResult:
        if not query or not query.strip():
            return QueryValidationResult(False, "Query string cannot be empty")

        dangerous_tokens = ["DROP", "DELETE", "TRUNCATE", "--", "ALTER", "UPDATE", "INSERT", "CREATE"]
        if any(token in query.upper() for token in dangerous_tokens):
            return QueryValidationResult(False, "Query contains potentially dangerous operations")

        return QueryValidationResult(True)

    def execute_metadata_query(self, query: str) -> List[Tuple]:
        """Execute metadata query with connection reuse."""
        try:
            validation_result = self._validate_query(query)
            if not validation_result.is_valid:
                raise ValueError(validation_result.error_message)

            with self.db_manager.get_connection() as connection:
                cursor = connection.cursor()

                logger.debug(f"Executing query: {query[:100]}...")
                initial_results = execute_query(cursor, query)

                if not initial_results:
                    logger.warning("No results found")
                    return []

                paper_ids = [result[0] for result in initial_results]
                paper_ids_clause = create_paper_ids_clause(paper_ids)
                papers_query = get_papers_query(paper_ids_clause)

                return execute_query(cursor, papers_query) or []

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
