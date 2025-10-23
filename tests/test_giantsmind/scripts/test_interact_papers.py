from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from giantsmind.core.models import MetadataResult, ParsedElements
from giantsmind.core.process_results import REQUIRED_METADATA_KEYS
from giantsmind.scripts.interact_papers import *


@pytest.fixture
def sample_parsed_elements():
    return ParsedElements(
        metadata_search="papers about machine learning",
        content_search="neural networks",
        general_knowledge="deep learning basics",
    )


def test_get_user_question(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "What is machine learning?")
    assert get_user_question() == "What is machine learning?"


def test_parse_user_question_success():
    with patch("giantsmind.agents.question_parsing.create_default_parser") as mock_parser:
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_question.return_value = {
            "metadata_search": "test metadata",
            "content_search": "test content",
            "general_knowledge": "test knowledge",
        }
        mock_parser.return_value = mock_parser_instance

        result = parse_user_question("What is machine learning?")

        # Better assertions for dataclass
        assert isinstance(result, ParsedElements)
        assert result.metadata_search == "test metadata"
        assert result.content_search == "test content"
        assert result.general_knowledge == "test knowledge"
        assert bool(result) is True  # Test the __bool__ method


def test_parse_user_question_error():
    with patch("giantsmind.agents.question_parsing.create_default_parser") as mock_parser:
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_question.side_effect = ValueError("Test error")
        mock_parser.return_value = mock_parser_instance

        result = parse_user_question("Invalid question")
        assert isinstance(result, ParsedElements)
        assert not result


# Add a new test for empty ParsedElements
def test_parse_user_question_empty():
    result = ParsedElements()
    assert bool(result) is False
    assert result.metadata_search is None
    assert result.content_search is None
    assert result.general_knowledge is None


def test_display_parsed_elements(capsys, sample_parsed_elements):
    display_parsed_elements(sample_parsed_elements)
    captured = capsys.readouterr()
    assert "Metadata search: papers about machine learning" in captured.out
    assert "Content search: neural networks" in captured.out


def test_modify_parsed_elements(monkeypatch, sample_parsed_elements):
    # Need 3 sets of inputs for metadata_search, content_search, and general_knowledge
    inputs = iter(["y", "new metadata", "n", "n", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    modified = modify_parsed_elements(sample_parsed_elements)
    assert modified.metadata_search == "new metadata"
    assert modified.content_search == "neural networks"
    assert modified.general_knowledge == "deep learning basics"


def test_get_metadata():
    with patch("giantsmind.agents.sql.get_sql_query") as mock_sql_query, patch(
        "giantsmind.agents.sql.metadata_query"
    ) as mock_metadata_query:

        mock_sql_query.return_value = "SELECT * FROM papers"
        mock_metadata_query.return_value = [{"paper_id": "123", "title": "Test", "authors": "Test"}]

        results = get_metadata("test query", "test_collection")
        assert isinstance(results, list)
        assert isinstance(results[0], MetadataResult)
        assert results[0].paper_id == "123"


def test_content_search():
    with patch("giantsmind.vector_db.search.execute_content_search") as mock_search:
        mock_search.return_value = ([Document(page_content="test content")], [0.8])
        metadata_results = [MetadataResult(paper_id="123", title="Test", authors="Test")]

        docs, scores = content_search("test query", metadata_results)
        assert isinstance(docs, list)
        assert isinstance(scores, list)
        assert len(docs) == 1
        assert len(scores) == 1


def test_answer_question():
    with patch("giantsmind.agents.answering.invoke") as mock_invoke:
        mock_invoke.return_value = "This is the answer"

        result = answer_question("test question", "test context")
        assert result == "This is the answer"


def test_print_results(capsys):
    print_results("Test answer")
    captured = capsys.readouterr()
    assert "Answer: Test answer" in captured.out


def test_one_question_chain():
    with patch("giantsmind.scripts.interact_papers.prompt_question") as mock_prompt, patch(
        "giantsmind.scripts.interact_papers.get_metadata"
    ) as mock_metadata, patch("giantsmind.scripts.interact_papers.content_search") as mock_content, patch(
        "giantsmind.scripts.interact_papers.answer_question"
    ) as mock_answer, patch(
        "giantsmind.scripts.interact_papers.print_results"
    ) as mock_print:

        mock_prompt.return_value = (
            "test question",
            ParsedElements(metadata_search="test", content_search="test", general_knowledge="test"),
        )
        metadata = dict(
            paper_id="123", title="Test", authors="Test", journal="Test", publication_date="2021-01-01"
        )
        mock_metadata.return_value = [MetadataResult(**metadata)]
        mock_doc = Document(page_content="test")
        mock_doc.metadata = metadata
        mock_content.return_value = [mock_doc]
        mock_answer.return_value = "Test answer"

        one_question_chain("test_collection")

        mock_prompt.assert_called_once()
        mock_metadata.assert_called_once()
        mock_content.assert_called_once()
        mock_answer.assert_called_once()
        mock_print.assert_called_once()
