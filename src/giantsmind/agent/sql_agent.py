from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Callable, Protocol, Dict, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)

import giantsmind.database as db_module
import giantsmind.database.config as db_config
import giantsmind.database.query as db_query
from giantsmind.utils import utils
from giantsmind.utils.logging import logger
from giantsmind.utils import local
from sqlite3 import Connection, Cursor
import sqlite3

# Constants
SQL_SYSTEM_MESSAGE_PATH = Path(__file__).parent / "messages" / "sql_system_message.txt"
DEFAULT_MODEL = "claude-3-5-sonnet-latest"
NO_QUERY = "NO_QUERY"
SQL_PREFIX = "SQL: "


@dataclass
class DatabaseConfig:
    path: str
    levenshtein_function: Callable[[str, str], int]


class DatabaseConnection(Protocol):
    def connect(self, database: str) -> Connection: ...


class SQLiteConnection:
    """Basic SQLite database connection."""

    def connect(self, database: str) -> Connection:
        return sqlite3.connect(database)


def _get_sql_schema() -> str:
    path = str(Path(db_module.__file__).parent / "schema.sql")
    with open(path, "r") as f:
        schema = f.read()
    return schema


def _sql_sys_msg(schema: str, collection_id: int) -> SystemMessage:
    with open(SQL_SYSTEM_MESSAGE_PATH, "r") as file:
        system_sql_template = file.read()
    system_sql = SystemMessage(system_sql_template.format(schema=schema, collection_id=collection_id))
    return system_sql


def _get_llm_model() -> BaseChatModel:
    utils.set_env_vars()
    model = ChatAnthropic(model=DEFAULT_MODEL)
    return model


def get_sql_query(user_message: str, collection_id: int = 1) -> str:
    schema = _get_sql_schema()
    system_sql = _sql_sys_msg(schema, collection_id)
    model = _get_llm_model()
    messages = [system_sql, HumanMessage(user_message)]
    answer = model.invoke(messages)
    return answer.content


def _preprocess_query(query: str) -> str | None:
    if query == NO_QUERY:
        return None
    if query.startswith(SQL_PREFIX):
        return query[len(SQL_PREFIX) :]
    logger.error(f"Cannot preprocess query: Invalid query {query}")
    raise ValueError(f"Wrong query {query}")


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


def execute_query(cursor: Cursor, query: str) -> List[Tuple]:
    """Execute SQL query and return results."""
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(f"Database error executing query: {query}\nError: {e}")
        raise


def setup_database_connection(
    connection: Connection, levenshtein_function: Callable[[str, str], int]
) -> None:
    """Setup database connection with required functions."""
    connection.create_function("levenshtein", 2, levenshtein_function)


def execute_metadata_query(
    query: str,
    preprocess_query: Callable[[str], str],
    process_results: Callable[[List[Tuple]], List[str]],
    db_connector: DatabaseConnection,
    config: DatabaseConfig,
) -> List[Dict[str, Any]]:
    """
    Execute metadata query with improved testability.

    Args:
        query: Raw query string
        preprocess_query: Function to preprocess the query
        process_results: Function to process the query results
        db_connector: Database connection interface
        config: Database configuration

    Returns:
        List[str]: Processed paper metadata
    """
    processed_query = preprocess_query(query)

    with db_connector.connect(config.path) as connection:
        setup_database_connection(connection, config.levenshtein_function)
        cursor = connection.cursor()

        # Get paper IDs
        initial_results = execute_query(cursor, processed_query)
        paper_ids = [tuple_[0] for tuple_ in initial_results]

        # Get paper metadata
        paper_ids_txt = create_paper_ids_clause(paper_ids)
        papers_query = get_papers_query(paper_ids_txt)
        papers_metadata = execute_query(cursor, papers_query)

    return process_results(papers_metadata)


def metadata_query(query: str) -> List[Dict[str, Any]]:
    dbconf = DatabaseConfig(path=db_config.DATABASE_URL, levenshtein_function=db_query.levenshtein_func)

    db_connector = SQLiteConnection()

    try:
        results = execute_metadata_query(
            query=query,
            preprocess_query=_preprocess_query,
            process_results=_process_results,
            db_connector=db_connector,
            config=dbconf,
        )
        return results
    except ValueError as e:
        # Handle specific errors from preprocess_query
        raise
    except sqlite3.Error as e:
        # Handle database-specific errors
        raise


if __name__ == "__main__":
    # Testing the SQL agent
    query = get_sql_query("Get all the papers from Emily Johnson", 1)
    query = get_sql_query("I want all the papers from Patel published in Nature", 1)
    query = get_sql_query("Has sarah williams published in 2022?", 1)
    execute_metadata_query(query)
    answer = get_sql_query(
        "Get all the papers that authors Robert Kennedy and Jennifer Lawrence published together in Plos comp biology published after 2010",
        1,
    )

    print(get_buffer_string([answer]))

    # Which papers are machine learning papers?
