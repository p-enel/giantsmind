import pytest
from datetime import date
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from giantsmind.database.paper_operations import (
    _get_paper,
    get_all_papers,
    _add_paper,
    get_papers,
    add_papers,
    _remove_paper,
    remove_papers,
    _add_chunks,
    add_chunks,
    PaperNotFoundError,
    PaperExistsError,
)
from giantsmind.database.schema import Paper, Author, Journal, ChunkIDs


@pytest.fixture
def mock_session():
    return Mock(spec=Session)


@pytest.fixture
def sample_metadata():
    return {
        "paper_id": "TEST:123",
        "journal": "Test Journal",
        "file_path": "/path/to/test.pdf",
        "publication_date": date(2023, 1, 1),
        "title": "Test Paper",
        "authors": ["John Doe", "Jane Smith"],
        "url": "https://example.com/test",
    }


def test_get_paper(mock_session):
    mock_paper = Paper(paper_id="TEST:123")
    mock_session.query().filter_by().one_or_none.return_value = mock_paper

    result = _get_paper(mock_session, "TEST:123")

    assert result == mock_paper
    mock_session.query.assert_called_with(Paper)
    mock_session.query().filter_by.assert_called_with(paper_id="TEST:123")


def test_get_all_papers(mock_session):
    mock_papers = [Paper(paper_id="TEST:1"), Paper(paper_id="TEST:2")]
    mock_session.query().all.return_value = mock_papers

    result = get_all_papers(mock_session)

    assert result == mock_papers
    mock_session.query.assert_called_with(Paper)
    mock_session.query().all.assert_called()


def test_add_paper_existing(mock_session, sample_metadata):
    mock_session.query().filter_by().one_or_none.return_value = Paper(paper_id="TEST:123")

    result = _add_paper(mock_session, sample_metadata)

    assert result is None
    mock_session.add.assert_not_called()
    mock_session.commit.assert_not_called()


def test_get_papers(mocker, mock_session):
    mocker.patch("giantsmind.database.paper_operations.Session", return_value=mock_session)
    mock_papers = [Paper(paper_id="TEST:1"), Paper(paper_id="TEST:2")]
    mock_session.query().filter_by().one_or_none.side_effect = mock_papers

    result = get_papers(["TEST:1", "TEST:2"])

    assert result == mock_papers


def test_get_papers_not_found(mocker, mock_session):
    mocker.patch("giantsmind.database.paper_operations.Session", return_value=mock_session)
    mock_session.query().filter_by().one_or_none.return_value = None

    with pytest.raises(PaperNotFoundError):
        get_papers(["NON:EXISTENT"])


def test_add_papers(mocker, mock_session, sample_metadata):
    mocker.patch("giantsmind.database.paper_operations.Session", return_value=mock_session)
    mock_session.query().filter_by().one_or_none.return_value = None

    result = add_papers([sample_metadata])

    assert len(result) == 1
    assert isinstance(result[0], Paper)


def test_remove_paper(mock_session):
    mock_paper = Paper(paper_id="TEST:123")

    _remove_paper(mock_session, mock_paper)

    mock_session.delete.assert_called_once_with(mock_paper)
    mock_session.commit.assert_called_once()


def test_remove_papers(mocker, mock_session):
    mocker.patch("giantsmind.database.paper_operations.Session", return_value=mock_session)
    mock_paper = Paper(paper_id="TEST:123")
    mock_session.query().filter_by().one_or_none.return_value = mock_paper

    remove_papers(["TEST:123"])

    mock_session.delete.assert_called_once_with(mock_paper)
    mock_session.commit.assert_called()


def test_remove_papers_not_found(mocker, mock_session):
    mocker.patch("giantsmind.database.paper_operations.Session", return_value=mock_session)
    mock_session.query().filter_by().one_or_none.return_value = None

    with pytest.raises(PaperNotFoundError):
        remove_papers(["NON:EXISTENT"])


def test_add_chunks(mock_session):
    mock_paper = Paper(paper_id="TEST:123")
    chunk_ids = ["CHUNK1", "CHUNK2"]

    _add_chunks(mock_session, chunk_ids, mock_paper)

    assert mock_session.add.call_count == 2
    mock_session.commit.assert_called_once()


def test_add_chunks_to_paper(mocker, mock_session):
    mocker.patch("giantsmind.database.paper_operations.Session", return_value=mock_session)
    mock_paper = Paper(paper_id="TEST:123")
    mock_session.query().filter_by().one_or_none.return_value = mock_paper
    chunk_ids = ["CHUNK1", "CHUNK2"]

    add_chunks(chunk_ids, "TEST:123")

    assert mock_session.add.call_count == 2
    mock_session.commit.assert_called_once()


def test_add_chunks_paper_not_found(mocker, mock_session):
    mocker.patch("giantsmind.database.paper_operations.Session", return_value=mock_session)
    mock_session.query().filter_by().one_or_none.return_value = None

    with pytest.raises(PaperNotFoundError):
        add_chunks(["CHUNK1"], "NON:EXISTENT")
