from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from giantsmind.agents import question_parsing as pa


@pytest.fixture
def mock_chat_anthropic():
    with patch("giantsmind.agents.question_parsing.ChatAnthropic") as mock:
        mock_instance = Mock()
        mock_instance.invoke.return_value.content = " test response "
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def default_config():
    return pa.ParserConfig(prompt_path=Path("/test/path"), model_name="test-model")


@pytest.fixture
def parser(default_config):
    return pa.ResponseParser(default_config)


@pytest.fixture
def mock_model_client():
    return Mock(spec=pa.ModelClient)


@pytest.fixture
def mock_generate_prompt():
    return Mock(return_value="test prompt")


@pytest.fixture
def mock_response_parser():
    return Mock(spec=pa.ResponseParser)


@pytest.fixture
def question_parser(default_config, mock_model_client, mock_generate_prompt, mock_response_parser):
    return pa.QuestionParser(default_config, mock_model_client, mock_generate_prompt, mock_response_parser)


@pytest.fixture
def mock_resources():
    with patch("giantsmind.agents.question_parsing.resources.files") as mock:
        mock_path = MagicMock()
        mock_path.__truediv__.return_value = Path("/mock/parsing_prompt.txt")
        mock.return_value = mock_path
        yield mock


def test_anthropic_client_init(mock_chat_anthropic):
    model_name = "test-model"
    client = pa.AnthropicClient(model_name)

    mock_chat_anthropic.assert_called_once_with(model=model_name)


def test_anthropic_client_get_response(mock_chat_anthropic):
    client = pa.AnthropicClient("test-model")
    prompt = "test prompt"

    response = client.get_response(prompt)

    mock_chat_anthropic.return_value.invoke.assert_called_once_with(prompt)
    assert response == "test response"  # Verify stripping whitespace


def test_parser_config_default_initialization():
    config = pa.ParserConfig(prompt_path=Path("/test/path"), model_name="test-model")

    assert config.prompt_path == Path("/test/path")
    assert config.model_name == "test-model"
    assert config.error_message == "Error:"
    assert config.search_prefixes == {
        "metadata_search": "Metadata Search: ",
        "content_search": "Content Search: ",
        "general_knowledge": "General Knowledge: ",
    }


def test_parser_config_custom_initialization():
    custom_prefixes = {"test1": "Test1: ", "test2": "Test2: "}

    config = pa.ParserConfig(
        prompt_path=Path("/test/path"),
        model_name="test-model",
        error_message="Custom Error:",
        search_prefixes=custom_prefixes,
    )

    assert config.prompt_path == Path("/test/path")
    assert config.model_name == "test-model"
    assert config.error_message == "Custom Error:"
    assert config.search_prefixes == custom_prefixes


def test_generate_prompt(tmp_path):
    # Create a temporary prompt file
    prompt_file = tmp_path / "test_prompt.txt"
    prompt_content = "Template text: {user_question}"
    prompt_file.write_text(prompt_content)

    question = "Test question"
    result = pa.generate_prompt(question, prompt_file)

    assert result == "Template text: Test question"


def test_generate_prompt_missing_file():
    with pytest.raises(FileNotFoundError):
        pa.generate_prompt("test", Path("nonexistent.txt"))


def test_generate_prompt_with_complex_template(tmp_path):
    prompt_file = tmp_path / "complex_prompt.txt"
    template = """Question: {user_question}
Processing request...
End of prompt"""
    prompt_file.write_text(template)

    question = "What is the meaning of life?"
    result = pa.generate_prompt(question, prompt_file)

    expected = """Question: What is the meaning of life?
Processing request...
End of prompt"""
    assert result == expected


def test_validate_response_with_error(parser):
    response = "Error: Something went wrong"
    result = parser.validate_response(response)
    assert result == {"error": response.strip()}


def test_validate_response_without_error(parser):
    response = "Valid response"
    result = parser.validate_response(response)
    assert result is None


def test_split_response(parser):
    response = "Line 1\nLine 2\n  Line 3  "
    result = parser.split_response(response)
    assert result == ["Line 1", "Line 2", "Line 3"]


def test_extract_search_value_valid(parser):
    line = "Metadata Search: value"
    result = parser.extract_search_value(line, "Metadata Search: ")
    assert result == "value"


def test_extract_search_value_none(parser):
    line = "Content Search: none"
    result = parser.extract_search_value(line, "Content Search: ")
    assert result is None


def test_extract_search_value_empty(parser):
    line = "General Knowledge: "
    result = parser.extract_search_value(line, "General Knowledge: ")
    assert result == ""


def test_extract_search_components_all_present(parser):
    lines = [
        "Metadata Search: author:Enel",
        "Content Search: reservoir computing",
        "General Knowledge: AI papers",
    ]
    result = parser.extract_search_components(lines)
    assert result == {
        "metadata_search": "author:Enel",
        "content_search": "reservoir computing",
        "general_knowledge": "AI papers",
    }


def test_extract_search_components_with_none(parser):
    lines = ["Metadata Search: none", "Content Search: reservoir computing", "General Knowledge: none"]
    result = parser.extract_search_components(lines)
    assert result == {
        "metadata_search": None,
        "content_search": "reservoir computing",
        "general_knowledge": None,
    }


def test_parse_question_success(
    question_parser, mock_model_client, mock_generate_prompt, mock_response_parser
):
    user_question = "test question"
    mock_model_client.get_response.return_value = "test response"
    mock_response_parser.validate_response.return_value = None
    expected_components = {"metadata_search": "test"}
    mock_response_parser.extract_search_components.return_value = expected_components

    result = question_parser.parse_question(user_question)

    assert result == expected_components
    mock_generate_prompt.assert_called_once_with(user_question, question_parser.config.prompt_path)
    mock_model_client.get_response.assert_called_once_with("test prompt")
    mock_response_parser.validate_response.assert_called_once_with("test response")
    mock_response_parser.split_response.assert_called_once_with("test response")


def test_parse_question_with_error(question_parser, mock_model_client, mock_response_parser):
    error_result = {"error": "test error"}
    mock_response_parser.validate_response.return_value = error_result

    result = question_parser.parse_question("test")

    assert result == error_result
    mock_response_parser.split_response.assert_not_called()
    mock_response_parser.extract_search_components.assert_not_called()


def test_create_default_parser(mock_chat_anthropic, mock_resources):
    parser = pa.create_default_parser()

    assert isinstance(parser, pa.QuestionParser)
    assert parser.config.model_name == "claude-3-5-sonnet-latest"
    assert parser.config.prompt_path.name == "parsing_prompt.txt"
    assert isinstance(parser.model_client, pa.AnthropicClient)
    assert isinstance(parser.response_parser, pa.ResponseParser)
    assert parser.generate_prompt == pa.generate_prompt

    mock_resources.assert_called_once_with("giantsmind.agents.resources.messages")
    mock_chat_anthropic.assert_called_once_with(model="claude-3-5-sonnet-latest")
