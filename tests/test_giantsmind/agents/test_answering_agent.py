from unittest.mock import MagicMock, mock_open, patch

from giantsmind.agents.answering import answer_question, generate_answering_prompt


def test_generate_answering_prompt():
    mock_prompt_template = "Here's your question: {user_question}\nHere's the context: {context}"
    mock_file = mock_open(read_data=mock_prompt_template)

    with patch("builtins.open", mock_file):
        result = generate_answering_prompt(
            user_question="What is AI?", context="AI is artificial intelligence"
        )

    expected = "Here's your question: What is AI?\nHere's the context: AI is artificial intelligence"
    assert result == expected
    mock_file.assert_called_once()


def test_answer_question():
    mock_prompt = "test prompt"
    mock_response = MagicMock()
    mock_response.content = "test response"

    mock_model = MagicMock()
    mock_model.invoke.return_value = mock_response

    mock_prompt_generator = MagicMock(return_value=mock_prompt)

    with patch("giantsmind.agents.answering.ChatAnthropic", return_value=mock_model):
        result = answer_question(
            user_question="test question", context="test context", prompt_generator=mock_prompt_generator
        )

    mock_prompt_generator.assert_called_once_with("test question", "test context")
    mock_model.invoke.assert_called_once_with(mock_prompt)
    assert result == "test response"
