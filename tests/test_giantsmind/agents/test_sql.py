from pathlib import Path
from unittest.mock import PropertyMock, mock_open, patch

import pytest
from langchain.schema import SystemMessage

from giantsmind.agents import sql as sa
from giantsmind.metadata_db.db_connection import DatabaseManager
from giantsmind.metadata_db.models import DatabaseFunction


@pytest.fixture(autouse=True)
def reset_db_manager():
    """Reset DatabaseManager singleton before each test."""
    DatabaseManager.reset()
    yield


def test_get_sql_schema_file_exists_false():
    """Test behavior when schema file doesn't exist"""
    with patch("importlib.resources.files") as mock_files:
        mock_files.return_value.joinpath.return_value.exists.return_value = False
        with pytest.raises(FileNotFoundError, match="Schema file not found at"):
            sa._get_sql_schema()


def test_get_sql_schema_logs_error():
    """Test that errors are logged properly"""
    with patch("importlib.resources.files") as mock_files, patch(
        "giantsmind.utils.logging.logger.error"
    ) as mock_logger:
        mock_files.return_value.joinpath.return_value.open.side_effect = IOError("Test error")
        with pytest.raises(IOError):
            sa._get_sql_schema()
        mock_logger.assert_called_once_with("Error reading schema file: Test error")


def test_get_sql_schema_strips_whitespace():
    """Test that whitespace is preserved in valid schema"""
    mock_schema = "  CREATE TABLE test  \n  (id INTEGER PRIMARY KEY);  \n"
    with patch("importlib.resources.files") as mock_files:
        mock_files.return_value.joinpath.return_value.exists.return_value = True
        mock_files.return_value.joinpath.return_value.open = mock_open(read_data=mock_schema)
        schema = sa._get_sql_schema()
        assert schema == mock_schema  # Original whitespace should be preserved
        assert schema.strip() == mock_schema.strip()  # But stripping should work


def test_sql_sys_msg_valid_input():
    """Test _sql_sys_msg with valid inputs"""
    schema = "CREATE TABLE test (id INTEGER PRIMARY KEY);"
    collection_id = 1
    template = "Schema: {schema}, Collection ID: {collection_id}"

    with patch("importlib.resources.files") as mock_files, patch(
        "giantsmind.agents.config.SQL_SYSTEM_MESSAGE_PATH", new_callable=PropertyMock
    ) as mock_path:
        mock_files.return_value.joinpath.return_value.read_text.return_value = template
        mock_path.return_value = "mock_template_path"

        message = sa._sql_sys_msg(schema, collection_id)
        assert isinstance(message, SystemMessage)
        assert message.content == "Schema: CREATE TABLE test (id INTEGER PRIMARY KEY);, Collection ID: 1"


def test_sql_sys_msg_invalid_schema():
    """Test _sql_sys_msg with invalid schema"""
    with pytest.raises(ValueError, match="Schema must be non-empty string"):
        sa._sql_sys_msg("", 1)


def test_sql_sys_msg_invalid_collection_id():
    """Test _sql_sys_msg with invalid collection_id"""
    schema = "CREATE TABLE test (id INTEGER PRIMARY KEY);"
    with pytest.raises(ValueError, match="collection_id must be non-negative integer"):
        sa._sql_sys_msg(schema, -1)


def test_sql_sys_msg_missing_template_fields():
    """Test _sql_sys_msg with missing template fields"""
    schema = "CREATE TABLE test (id INTEGER PRIMARY KEY);"
    collection_id = 1
    template = "Schema: {schema}"

    with patch("importlib.resources.files") as mock_files, patch(
        "giantsmind.agents.config.SQL_SYSTEM_MESSAGE_PATH", new_callable=PropertyMock
    ) as mock_path:
        mock_files.return_value.joinpath.return_value.read_text.return_value = template
        mock_path.return_value = "mock_template_path"

        with pytest.raises(ValueError, match="Template missing required fields"):
            sa._sql_sys_msg(schema, collection_id)


def test_sql_sys_msg_template_read_error():
    """Test _sql_sys_msg when template file read fails"""
    schema = "CREATE TABLE test (id INTEGER PRIMARY KEY);"
    collection_id = 1

    with patch("importlib.resources.files") as mock_files, patch(
        "giantsmind.utils.logging.logger.error"
    ) as mock_logger:
        mock_files.return_value.joinpath.return_value.read_text.side_effect = IOError("Test error")

        with pytest.raises(IOError):
            sa._sql_sys_msg(schema, collection_id)
        mock_logger.assert_called_once_with("Error reading template file: Test error")


def test_preprocess_query_valid():
    """Test _preprocess_query with valid SQL query"""
    query = "SQL: SELECT * FROM papers"
    assert sa._preprocess_query(query) == "SELECT * FROM papers"


def test_preprocess_query_with_whitespace():
    """Test _preprocess_query handles whitespace correctly"""
    query = "  SQL:    SELECT * FROM papers   "
    assert sa._preprocess_query(query) == "SELECT * FROM papers"


def test_preprocess_query_no_query():
    """Test _preprocess_query with NO_QUERY constant"""
    assert sa._preprocess_query(sa.agent_cfg.NO_QUERY) is None


def test_preprocess_query_invalid_type():
    """Test _preprocess_query with non-string input"""
    with pytest.raises(TypeError, match="Query must be a string"):
        sa._preprocess_query(123)


def test_preprocess_query_empty():
    """Test _preprocess_query with empty string"""
    with pytest.raises(ValueError, match="Query cannot be empty or whitespace"):
        sa._preprocess_query("")
    with pytest.raises(ValueError, match="Query cannot be empty or whitespace"):
        sa._preprocess_query("   ")


def test_preprocess_query_wrong_prefix():
    """Test _preprocess_query with wrong prefix"""
    with pytest.raises(ValueError, match="Query must start with 'SQL:'"):
        sa._preprocess_query("SELECT * FROM papers")


def test_preprocess_query_empty_after_prefix():
    """Test _preprocess_query with empty query after prefix"""
    with pytest.raises(ValueError, match="Query is empty after prefix removal"):
        sa._preprocess_query("SQL:   ")


def test_format_results_valid():
    """Test _format_results with valid input"""
    input_data = [
        ("Title 1", "Journal 1", "2023", "Author 1", "id1", "url1"),
        ("Title 2", "Journal 2", "2024", "Author 2", "id2", "url2"),
    ]
    expected = [
        {
            "title": "Title 1",
            "journal": "Journal 1",
            "publication_date": "2023",
            "authors": "Author 1",
            "paper_id": "id1",
            "url": "url1",
        },
        {
            "title": "Title 2",
            "journal": "Journal 2",
            "publication_date": "2024",
            "authors": "Author 2",
            "paper_id": "id2",
            "url": "url2",
        },
    ]
    assert sa._format_results(input_data) == expected


def test_format_results_empty_list():
    """Test _format_results with empty list"""
    assert sa._format_results([]) == []


def test_format_results_invalid_type():
    """Test _format_results with non-list input"""
    with pytest.raises(TypeError, match="Results must be a list"):
        sa._format_results("not a list")


def test_format_results_invalid_tuple_length():
    """Test _format_results with tuple of wrong length"""
    invalid_data = [("Title 1", "Journal 1", "2023")]  # Missing fields
    with pytest.raises(ValueError, match="Expected 6 fields, got 3"):
        sa._format_results(invalid_data)


def test_format_results_invalid_item_type():
    """Test _format_results with non-tuple item"""
    invalid_data = [["Title 1", "Journal 1", "2023", "Author 1", "id1", "url1"]]  # List instead of tuple
    with pytest.raises(TypeError, match="Expected tuple, got list"):
        sa._format_results(invalid_data)


def test_get_sql_query_valid_input():
    """Test get_sql_query with valid input"""
    schema = "CREATE TABLE test (id INTEGER PRIMARY KEY);"
    expected_query = "SELECT * FROM test"

    def mock_schema():
        return schema

    def mock_message(s, c):
        return SystemMessage(content="test")

    def mock_generator(messages):
        return expected_query

    result = sa.get_sql_query(
        "Find all records",
        schema_provider=mock_schema,
        message_creator=mock_message,
        query_generator=mock_generator,
    )
    assert result == expected_query


def test_get_sql_query_empty_message():
    """Test get_sql_query with empty message"""
    with pytest.raises(ValueError, match="user_message must be non-empty string"):
        sa.get_sql_query("")


def test_get_sql_query_invalid_message_type():
    """Test get_sql_query with non-string message"""
    with pytest.raises(ValueError, match="user_message must be non-empty string"):
        sa.get_sql_query(123)


def test_get_sql_query_invalid_collection_id():
    """Test get_sql_query with invalid collection_id"""
    with pytest.raises(ValueError, match="collection_id must be non-negative integer"):
        sa.get_sql_query("Find all records", collection_id=-1)


def test_get_sql_query_schema_provider_error():
    """Test get_sql_query when schema provider fails"""

    def failing_schema():
        raise ValueError("Schema error")

    with pytest.raises(ValueError, match="Schema error"):
        sa.get_sql_query("Find all records", schema_provider=failing_schema)


def test_get_sql_query_message_creator_error():
    """Test get_sql_query when message creator fails"""

    def mock_schema():
        return "schema"

    def failing_message(s, c):
        raise ValueError("Message error")

    with pytest.raises(ValueError, match="Message error"):
        sa.get_sql_query("Find all records", schema_provider=mock_schema, message_creator=failing_message)


def test_get_sql_query_generator_returns_none():
    """Test get_sql_query when generator returns None"""

    def mock_schema():
        return "schema"

    def mock_message(s, c):
        return SystemMessage(content="test")

    def mock_generator(messages):
        return None

    with pytest.raises(ValueError, match="Model returned NO_QUERY"):
        sa.get_sql_query(
            "Find all records",
            schema_provider=mock_schema,
            message_creator=mock_message,
            query_generator=mock_generator,
        )


@patch("giantsmind.agents.sql.logger")  # Patch the logger where it's used
def test_get_sql_query_logs_debug_and_info(mock_logger):
    """Test get_sql_query logging behavior"""
    test_query = "SELECT * FROM test"
    test_message = "Find all records"

    def mock_schema():
        return "schema"

    def mock_message(s, c):
        return SystemMessage(content="test")

    def mock_generator(messages):
        return test_query

    result = sa.get_sql_query(
        test_message,
        schema_provider=mock_schema,
        message_creator=mock_message,
        query_generator=mock_generator,
    )

    assert result == test_query
    mock_logger.debug.assert_called_once_with(f"Requesting SQL for message: {test_message}")
    mock_logger.info.assert_called_once_with(f"Generated SQL query: {test_query}")


def test_metadata_query_valid():
    """Test metadata_query with valid input"""
    test_query = "SQL: SELECT * FROM papers"
    mock_results = [("Title", "Journal", "2023", "Author", "id1", "url1")]

    def mock_preprocess(q):
        return "SELECT * FROM papers"

    def mock_format(r):
        return [
            {
                "title": "Title",
                "journal": "Journal",
                "publication_date": "2023",
                "authors": "Author",
                "paper_id": "id1",
                "url": "url1",
            }
        ]

    class MockExecutor:
        def execute_metadata_query(self, query):
            assert query == "SELECT * FROM papers"
            return mock_results

    result = sa.metadata_query(
        test_query,
        query_executor=MockExecutor(),
        preprocess_query=mock_preprocess,
        format_results=mock_format,
    )
    assert result == [
        {
            "title": "Title",
            "journal": "Journal",
            "publication_date": "2023",
            "authors": "Author",
            "paper_id": "id1",
            "url": "url1",
        }
    ]


def test_metadata_query_no_query():
    """Test metadata_query with NO_QUERY constant"""

    def mock_preprocess(q):
        return None

    result = sa.metadata_query(
        sa.agent_cfg.NO_QUERY, preprocess_query=mock_preprocess, format_results=lambda x: []
    )
    assert result == []


def test_metadata_query_empty():
    """Test metadata_query with empty string"""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        sa.metadata_query("", preprocess_query=lambda x: x, format_results=lambda x: x)


def test_metadata_query_db_not_found():
    """Test metadata_query when database file doesn't exist"""
    test_path = "/path/to/nonexistent.db"
    test_config = sa.DatabaseConfig(path=test_path, db_functions=[])

    def mock_preprocess(q):
        return "SELECT * FROM papers"

    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="Database not found at"):
            sa.metadata_query(
                "SQL: SELECT * FROM papers",
                db_config=test_config,
                preprocess_query=mock_preprocess,
                format_results=lambda x: x,
            )


def test_metadata_query_execution_error():
    """Test metadata_query when query execution fails"""

    def mock_preprocess(q):
        return "SELECT * FROM papers"

    class FailingExecutor:
        def execute_metadata_query(self, _):
            raise Exception("Database error")

    with pytest.raises(Exception, match="Database error"):
        sa.metadata_query(
            "SQL: SELECT * FROM papers",
            query_executor=FailingExecutor(),
            preprocess_query=mock_preprocess,
            format_results=lambda x: x,
        )


@pytest.mark.parametrize(
    "results,expected",
    [
        ([], []),
        (
            [("Title", "Journal", "2023", "Author", "id1", "url1")],
            [
                {
                    "title": "Title",
                    "journal": "Journal",
                    "publication_date": "2023",
                    "authors": "Author",
                    "paper_id": "id1",
                    "url": "url1",
                }
            ],
        ),
        (
            [("T1", "J1", "2023", "A1", "id1", "url1"), ("T2", "J2", "2024", "A2", "id2", "url2")],
            [
                {
                    "title": "T1",
                    "journal": "J1",
                    "publication_date": "2023",
                    "authors": "A1",
                    "paper_id": "id1",
                    "url": "url1",
                },
                {
                    "title": "T2",
                    "journal": "J2",
                    "publication_date": "2024",
                    "authors": "A2",
                    "paper_id": "id2",
                    "url": "url2",
                },
            ],
        ),
    ],
)
def test_metadata_query_different_results(results, expected):
    """Test metadata_query with different result sets"""

    def mock_preprocess(q):
        return "SELECT * FROM papers"

    class MockExecutor:
        def execute_metadata_query(self, _):
            return results

    result = sa.metadata_query(
        "SQL: SELECT * FROM papers",
        query_executor=MockExecutor(),
        preprocess_query=mock_preprocess,
        format_results=sa._format_results,
    )
    assert result == expected


def test_create_query_executor_default_config():
    """Test _create_query_executor with default configuration"""
    with patch("pathlib.Path.exists", return_value=True), patch(
        "giantsmind.metadata_db.config.DEFAULT_DATABASE_PATH", new=Path("/mock/db.sqlite")
    ):
        executor = sa._create_query_executor()
        assert executor is not None
        assert executor.db_manager.config.path == Path("/mock/db.sqlite")
        assert len(executor.db_manager.config.db_functions) == 2


def test_create_query_executor_custom_config_and_connection():
    """Test _create_query_executor with custom configuration"""

    class MockConnection:
        def __init__(self, path):
            self.path = path

    mock_function = DatabaseFunction("test_func", 1, lambda x: x)
    custom_config = sa.DatabaseConfig(path=Path("/custom/path.db"), db_functions=[mock_function])

    with patch("pathlib.Path.exists", return_value=True):
        executor = sa._create_query_executor(custom_config, connection_cls=MockConnection)
        assert executor.db_manager.connection_cls == MockConnection
        assert executor.db_manager.config == custom_config
        assert executor.db_manager.config.path == Path("/custom/path.db")
        assert len(executor.db_manager.config.db_functions) == 1
        assert isinstance(executor.db_manager.config.db_functions[0], DatabaseFunction)


def test_create_query_executor_missing_db():
    """Test _create_query_executor with non-existent database"""

    class MockConnection:
        def __init__(self, path):
            self.path = path

    mock_config = sa.DatabaseConfig(path=Path("/nonexistent/db.sqlite"), db_functions=[])

    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="Database not found at"):
            sa._create_query_executor(db_config=mock_config, connection_cls=MockConnection)
