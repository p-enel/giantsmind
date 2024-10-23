import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from unittest.mock import patch, MagicMock
from datetime import date

from giantsmind.database.utils import (
    _get_unique_values,
    _get_distance,
    _sort,
    _paper_to_dict,
    _authors2str,
    print_papers,
    print_papers_from_collection,
)
from giantsmind.database.schema import Paper, Author, Journal


@pytest.fixture
def mock_session():
    return MagicMock(spec=Session)


@pytest.fixture
def mock_engine():
    return MagicMock(spec=create_engine)


def test_get_unique_values():
    class MockRow:
        def __init__(self, value):
            self.column = value

    rows = [MockRow("a"), MockRow("b"), MockRow("a"), MockRow("c")]
    result = _get_unique_values(rows, "column")
    assert set(result) == {"a", "b", "c"}


@pytest.mark.parametrize(
    "values,search_term,expected",
    [
        (["apple", "banana", "cherry"], "aple", [1, 5, 6]),
        (["cat", "dog", "elephant"], "dot", [2, 1, 7]),
    ],
)
def test_get_distance(values, search_term, expected):
    result = _get_distance(values, search_term)
    assert result == expected


def test_sort():
    values = ["cherry", "apple", "banana"]
    distances = [2, 1, 3]
    sorted_values, sorted_distances = _sort(values, distances)
    assert sorted_values == ["apple", "cherry", "banana"]
    assert sorted_distances == [1, 2, 3]


def test_paper_to_dict():
    mock_paper = MagicMock(spec=Paper)
    mock_paper.__dict__ = {
        "title": "Test Paper",
        "publication_date": date(2022, 1, 1),
        "_sa_instance_state": "to_be_removed",
    }

    result = _paper_to_dict(mock_paper)

    assert "title" in result
    assert "publication_date" in result
    assert "_sa_instance_state" not in result


def test_authors2str(mocker):
    # Create two separate mock Author objects
    mock_author1 = mocker.Mock(spec=Author)
    mock_author2 = mocker.Mock(spec=Author)

    # Set the name attribute for each mock
    mock_author1.name = "John Doe"
    mock_author2.name = "Jane Smith"

    result = _authors2str([mock_author1, mock_author2])

    assert result == "John Doe, Jane Smith"
