from unittest.mock import patch

import pytest
from langchain_core.documents.base import Document

from giantsmind.core.models import MetadataResult, ParsedElements
from giantsmind.scripts import interact_papers


@pytest.fixture
def mock_user_input():
    return "What method is used to multimodal fusion in papers published before 2022?"


@pytest.fixture
def mock_parsed_elements():
    return ParsedElements(
        metadata_search="Papers published before 2022",
        content_search="method used for multimodal fusion",
        general_knowledge=None,
    )


metadata_results = MetadataResult(
    paper_id="123",
    title="Early paper on multimodal fusion.",
    authors="John Doe",
    publication_date="2021-01-01",
    journal="Journal of AI",
)


@pytest.fixture
def mock_metadata_results():
    return [metadata_results]


@pytest.fixture
def mock_content_results():
    return [
        Document(
            page_content="Neural networks are...",
            metadata=metadata_results.__dict__,
        )
    ]


def test_full_interaction_chain(
    mock_user_input, mock_parsed_elements, mock_metadata_results, mock_content_results
):
    with patch("giantsmind.scripts.interact_papers.get_user_question") as mock_user_input, patch(
        "giantsmind.agents.question_parsing.create_default_parser"
    ) as mock_parser, patch("giantsmind.agents.sql.get_sql_query") as mock_sql_query, patch(
        "giantsmind.agents.sql.metadata_query"
    ) as mock_metadata_query, patch(
        "giantsmind.vector_db.search.execute_content_search"
    ) as mock_content_search, patch(
        "giantsmind.agents.answering.invoke"
    ) as mock_answering, patch(
        "giantsmind.scripts.interact_papers.handle_user_modifications", return_value=mock_parsed_elements
    ) as mock_handle_modifications:

        # Configure mocks
        mock_user_input.return_value = "What papers discuss neural networks?"
        mock_parser.return_value.parse_question.return_value = mock_parsed_elements
        mock_sql_query.return_value = "SELECT * FROM papers WHERE..."
        mock_metadata_query.return_value = [vars(result) for result in mock_metadata_results]
        mock_content_search.return_value = mock_content_results
        mock_answering.return_value = "Neural networks are fundamental to deep learning."

        # Execute the main chain
        interact_papers.one_question_chain("test_collection")

        # Verify interactions
        mock_parser.assert_called_once()
        mock_sql_query.assert_called_once()
        mock_metadata_query.assert_called_once()
        mock_content_search.assert_called_once()
        mock_answering.assert_called_once()
        mock_handle_modifications.assert_called_once()


def test_error_handling_in_parsing():
    with patch("builtins.input", return_value="Invalid question"), patch(
        "giantsmind.agents.question_parsing.create_default_parser"
    ) as mock_parser:

        mock_parser.return_value.parse_question.side_effect = ValueError("Parsing failed")

        # This should handle the error gracefully
        result = interact_papers.parse_user_question("Invalid question")
        assert isinstance(result, ParsedElements)
        assert not result
