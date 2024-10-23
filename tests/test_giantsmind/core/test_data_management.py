import pytest
from unittest.mock import patch
from giantsmind.core.data_management import (
    load_markdown_paper,
    convert_pdf_path_to_md_fname,
    get_paper_txts_from_collection_id,
    combine_metadata_and_txt,
    add_separator_to_txts,
    get_context_from_collection,
)


@pytest.fixture
def mock_local_data_path(tmp_path):
    with patch("giantsmind.utils.local.get_local_data_path", return_value=str(tmp_path)):
        yield tmp_path


def test_load_markdown_paper(tmp_path):
    test_content = "# Test Markdown\nThis is a test."
    test_file = tmp_path / "test.md"
    test_file.write_text(test_content)

    result = load_markdown_paper(str(test_file))
    assert result == test_content


def test_convert_pdf_path_to_md_fname(mock_local_data_path):
    pdf_path = "/path/to/paper.pdf"
    expected_md_path = mock_local_data_path / "parsed_docs" / "paper.md"

    result = convert_pdf_path_to_md_fname(pdf_path)
    assert result == str(expected_md_path)


@patch("giantsmind.database.operations.get_paper_paths_from_collection_id")
@patch("giantsmind.core.data_management.load_markdown_paper")
@patch("giantsmind.core.data_management.convert_pdf_path_to_md_fname")
def test_get_paper_txts_from_collection_id(mock_convert, mock_load, mock_get_paths):
    mock_get_paths.return_value = ["/path/to/paper1.pdf", "/path/to/paper2.pdf"]
    mock_convert.side_effect = ["/path/to/paper1.md", "/path/to/paper2.md"]
    mock_load.side_effect = ["Content of paper 1", "Content of paper 2"]

    result = get_paper_txts_from_collection_id(1)
    assert result == ["Content of paper 1", "Content of paper 2"]
    mock_get_paths.assert_called_once_with(1)


def test_combine_metadata_and_txt():
    metadata = {
        "title": "Test Paper",
        "authors": "John Doe; Jane Smith",
        "journal": "Test Journal",
        "publication_date": "2023-01-01",
        "paper_id": "TEST123",
    }
    paper_txt = "This is the content of the paper."

    result = combine_metadata_and_txt(metadata, paper_txt)
    assert "<title> Test Paper </title>" in result
    assert "<authors> John Doe; Jane Smith </authors>" in result
    assert "<journal> Test Journal </journal>" in result
    assert "<publication date> 2023-01-01 </publication date>" in result
    assert "<paper ID> TEST123 </paper ID>" in result
    assert "<body> This is the content of the paper. </body>" in result


def test_add_separator_to_txts():
    txts = ["Text 1", "Text 2", "Text 3"]
    result = add_separator_to_txts(txts)
    expected = "Text 1\n" + "-" * 80 + "\nText 2\n" + "-" * 80 + "\nText 3\n" + "-" * 80
    assert result == expected


@patch("giantsmind.database.operations.get_collection_id")
@patch("giantsmind.core.data_management.get_paper_txts_from_collection_id")
@patch("giantsmind.database.operations.get_metadata_from_collection_id")
@patch("giantsmind.core.data_management.combine_metadata_and_txt")
@patch("giantsmind.core.data_management.add_separator_to_txts")
def test_get_context_from_collection(
    mock_add_separator, mock_combine, mock_get_metadata, mock_get_txts, mock_get_id
):
    mock_get_id.return_value = 1
    mock_get_txts.return_value = ["Paper 1 content", "Paper 2 content"]
    mock_get_metadata.return_value = [
        {
            "title": "Paper 1",
            "authors": "Author 1",
            "journal": "Journal 1",
            "publication_date": "2023-01-01",
            "paper_id": "ID1",
        },
        {
            "title": "Paper 2",
            "authors": "Author 2",
            "journal": "Journal 2",
            "publication_date": "2023-02-01",
            "paper_id": "ID2",
        },
    ]
    mock_combine.side_effect = ["Combined Paper 1", "Combined Paper 2"]
    mock_add_separator.return_value = "Final Context"

    result = get_context_from_collection("Test Collection")
    assert result == "Final Context"
    mock_get_id.assert_called_once_with("Test Collection")
    mock_get_txts.assert_called_once_with(1)
    mock_get_metadata.assert_called_once_with(1)
    assert mock_combine.call_count == 2
    mock_add_separator.assert_called_once_with(["Combined Paper 1", "Combined Paper 2"])
