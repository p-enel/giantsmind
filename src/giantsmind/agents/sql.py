from importlib import resources
from pathlib import Path
from string import Formatter
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)

from giantsmind.agents import config as agent_cfg
from giantsmind.metadata_db import config as db_cfg_module
from giantsmind.metadata_db import database_functions as db_functions
from giantsmind.metadata_db.db_connection import SQLiteConnection
from giantsmind.metadata_db.models import DatabaseConfig, DatabaseConnection
from giantsmind.metadata_db.operations import collection_operations as col_ops
from giantsmind.metadata_db.query_executor import QueryExecutor
from giantsmind.utils.logging import logger

load_dotenv()


class PaperResult(NamedTuple):
    title: str
    journal: str
    publication_date: str
    authors: str
    paper_id: str
    url: str


def _get_sql_schema() -> str:
    """Get SQL schema from resources file.

    Returns:
        str: The SQL schema string

    Raises:
        FileNotFoundError: If schema.sql file is not found
        ValueError: If schema file is empty
    """
    try:
        schema_path = resources.files("giantsmind.metadata_db.resources").joinpath("schema.sql")
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found at {schema_path}")

        with schema_path.open("r") as f:
            schema = f.read()

        if not schema.strip():
            raise ValueError("Schema file is empty")

        return schema

    except (OSError, IOError) as e:
        logger.error(f"Error reading schema file: {e}")
        raise


def _sql_sys_msg(schema: str, collection_id: int) -> SystemMessage:
    try:
        if not schema or not isinstance(schema, str):
            raise ValueError("Schema must be non-empty string")

        if not isinstance(collection_id, int) or collection_id < 0:
            raise ValueError("collection_id must be non-negative integer")

        template = (
            resources.files("giantsmind.agents.resources")
            .joinpath(agent_cfg.SQL_SYSTEM_MESSAGE_PATH)
            .read_text()
        )

        # Validate template placeholders
        required_fields = {"schema", "collection_id"}
        template_fields = {fname for _, fname, _, _ in Formatter().parse(template) if fname}

        if not required_fields.issubset(template_fields):
            raise ValueError(f"Template missing required fields: {required_fields - template_fields}")

        return SystemMessage(template.format(schema=schema, collection_id=collection_id))

    except (OSError, IOError) as e:
        logger.error(f"Error reading template file: {e}")
        raise


def _get_llm_model() -> BaseChatModel:
    return ChatAnthropic(model=agent_cfg.DEFAULT_MODEL)


def _query_generator(messages: List[BaseMessage]) -> str:
    return _get_llm_model().invoke(messages).content.strip()


def _preprocess_query(query: str) -> str | None:
    if not isinstance(query, str):
        raise TypeError("Query must be a string")

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty or whitespace")

    if query == agent_cfg.NO_QUERY:
        return None

    if query.startswith(agent_cfg.SQL_PREFIX):
        processed = query[len(agent_cfg.SQL_PREFIX) :].strip()
        if not processed:
            raise ValueError("Query is empty after prefix removal")
        return processed

    logger.error(f"Invalid query format: {query}")
    raise ValueError(f"Query must start with '{agent_cfg.SQL_PREFIX}'")


def _format_results(results: List[Tuple[Any, ...]]) -> List[Dict[str, str]]:
    if not isinstance(results, list):
        raise TypeError("Results must be a list")

    expected_length = len(PaperResult._fields)

    processed = []
    for result in results:
        if not isinstance(result, tuple):
            raise TypeError(f"Expected tuple, got {type(result).__name__}")

        if len(result) != expected_length:
            raise ValueError(f"Expected {expected_length} fields, got {len(result)}")

        try:
            paper = PaperResult(*result)
            processed.append(paper._asdict())
        except Exception as e:
            logger.error(f"Failed to process result {result}: {e}")
            raise

    return processed


def get_sql_query(
    user_message: str,
    schema_provider: Callable[[], str] = _get_sql_schema,
    message_creator: Callable[[str, int], SystemMessage] = _sql_sys_msg,
    query_generator: Callable[[List[BaseMessage]], str] = _query_generator,
    collection_name: str = db_cfg_module.DEFAULT_COLLECTION,
) -> str:
    """Generate SQL query from user message using LLM.

    Args:
        user_message: User's natural language query
        schema_provider: Function to get SQL schema, defaults to _get_sql_schema
        message_creator: Function to create system message, defaults to _sql_sys_msg
        query_generator: Function to generate SQL query, defaults to LLM model invoke
        logger: Logger instance, defaults to module logger
        collection_id: Target collection ID, defaults to 1

    Returns:
        Generated SQL query string

    Raises:
        ValueError: If inputs invalid or response malformed
        TypeError: If input types incorrect
    """
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("user_message must be non-empty string")

    if not isinstance(collection_name, str):
        raise ValueError("collection_id must be non-negative integer")

    collection_id = col_ops.get_collection_id(collection_name)
    try:
        schema = schema_provider()
        system_sql = message_creator(schema, collection_id)
        messages = [system_sql, HumanMessage(user_message)]

        logger.debug(f"Requesting SQL for message: {user_message}")
        sql_query = query_generator(messages)

        if sql_query is None:
            raise ValueError("Model returned NO_QUERY")

        logger.info(f"Generated SQL query: {sql_query}")
        return sql_query

    except Exception as e:
        logger.error(f"Error generating SQL query: {e}")
        raise


def metadata_query(
    query: str,
    *,
    query_executor: QueryExecutor | None = None,
    db_config: DatabaseConfig | None = None,
    connection_class: Type[DatabaseConnection] = SQLiteConnection,
    preprocess_query: Callable[[str], str | None] = _preprocess_query,
    format_results: Callable[[List[Tuple]], List[Dict[str, Any]]] = _format_results,
) -> List[Dict[str, Any]]:
    """Execute metadata query and process results.

    Args:
        query: SQL query string to execute
        query_executor: Optional preconfigured QueryExecutor
        db_config: Optional database configuration
        connection_class: Database connection class, defaults to SQLite
        preprocess_query: Query preprocessing function
        process_results: Results processing function

    Returns:
        List of dictionaries containing query results

    Raises:
        TypeError: If query is not a string
        ValueError: If query is empty
        FileNotFoundError: If database not found
    """
    if not isinstance(query, str):
        raise TypeError("Query must be a string")

    if not query.strip():
        raise ValueError("Query cannot be empty")

    processed_query = preprocess_query(query)
    if processed_query is None:
        return []

    try:
        executor = query_executor or _create_query_executor(db_config, connection_class)
        query_results = executor.execute_metadata_query(processed_query)
        results = format_results(query_results)
        if not isinstance(results, list):
            raise TypeError("process_results must return a list")
        return results

    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise


def _create_query_executor(
    db_config: DatabaseConfig | None = None,
    connection_cls: Type[DatabaseConnection] = SQLiteConnection,
) -> QueryExecutor:
    """Create QueryExecutor with provided or default configuration."""
    config = db_config or DatabaseConfig(
        path=db_cfg_module.DEFAULT_DATABASE_PATH,
        db_functions=[db_functions.levenshtein_func, db_functions.author_name_distance_func],
    )

    db_path = Path(config.path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    return QueryExecutor(connection_cls, config)


if __name__ == "__main__":
    # Testing the SQL agent
    answer = get_sql_query(
        "Get all the papers that authors Robert Kennedy and Jennifer Lawrence published together in Plos comp biology published after 2010",
        1,
    )

    print(get_buffer_string([answer]))

    # Which papers are machine learning papers?
