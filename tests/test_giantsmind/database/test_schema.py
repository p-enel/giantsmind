import pytest
from sqlalchemy import inspect, create_engine
from giantsmind.database.schema import Base


engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(engine)


@pytest.fixture(scope="module")
def connection():
    """Create a new database connection for the tests."""
    connection = engine.connect()
    yield connection
    connection.close()


def test_tables_exist(connection):
    inspector = inspect(connection)
    tables = inspector.get_table_names()
    assert "papers" in tables
    assert "collections" in tables
    assert "paper_collection" in tables


def test_papers_columns(connection):
    inspector = inspect(connection)
    columns = inspector.get_columns("papers")
    expected_columns = ["paper_id", "journal", "file_path", "publication_date", "title", "author", "url"]
    actual_columns = [column["name"] for column in columns]
    assert all(column in actual_columns for column in expected_columns)


def test_collections_columns(connection):
    inspector = inspect(connection)
    columns = inspector.get_columns("collections")
    expected_columns = ["collection_id", "name"]
    actual_columns = [column["name"] for column in columns]
    assert all(column in actual_columns for column in expected_columns)


def test_paper_collection_columns(connection):
    inspector = inspect(connection)
    columns = inspector.get_columns("paper_collection")
    expected_columns = ["paper_id", "collection_id"]
    actual_columns = [column["name"] for column in columns]
    assert all(column in actual_columns for column in expected_columns)
