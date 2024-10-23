import pytest
from sqlalchemy.orm import Session
from giantsmind.database.journal_operations import (
    _get_journal,
    _get_journal_from_id,
    _get_journal_id_from_paper_id,
)
from giantsmind.database.schema import Journal


@pytest.fixture
def mock_session(mocker):
    return mocker.Mock(spec=Session)


@pytest.fixture
def sample_journal():
    return Journal(journal_id="J001", name="Sample Journal")


def test_get_journal(mock_session, sample_journal):
    mock_session.query.return_value.filter_by.return_value.one_or_none.return_value = sample_journal
    result = _get_journal(mock_session, "Sample Journal")
    assert result == sample_journal
    mock_session.query.assert_called_once_with(Journal)
    mock_session.query.return_value.filter_by.assert_called_once_with(name="Sample Journal")


def test_get_journal_not_found(mock_session):
    mock_session.query.return_value.filter_by.return_value.one_or_none.return_value = None
    result = _get_journal(mock_session, "Nonexistent Journal")
    assert result is None


def test_get_journal_from_id(mock_session, sample_journal):
    mock_session.query.return_value.filter_by.return_value.one_or_none.return_value = sample_journal
    result = _get_journal_from_id(mock_session, "J001")
    assert result == sample_journal
    mock_session.query.assert_called_once_with(Journal)
    mock_session.query.return_value.filter_by.assert_called_once_with(journal_id="J001")


def test_get_journal_from_id_not_found(mock_session):
    mock_session.query.return_value.filter_by.return_value.one_or_none.return_value = None
    result = _get_journal_from_id(mock_session, "J999")
    assert result is None


@pytest.mark.parametrize(
    "paper_id, expected_journal_id",
    [
        ("DOI:10.1234/journal.paper123", "10.1234"),
        ("ARXIV:2104.12345", "arXiv"),
        ("OTHER:1234.5678", None),
    ],
)
def test_get_journal_id_from_paper_id(paper_id, expected_journal_id):
    result = _get_journal_id_from_paper_id(paper_id)
    assert result == expected_journal_id
