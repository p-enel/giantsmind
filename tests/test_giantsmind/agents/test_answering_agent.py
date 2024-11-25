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
    # Mock the ChatAnthropic instance and response
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "This is a test answer"
    mock_instance.invoke.return_value = mock_response

    with patch("giantsmind.agents.answering.generate_answering_prompt") as mock_generate_prompt, patch(
        "giantsmind.agents.answering.ChatAnthropic"
    ) as mock_chat_anthropic_class:
        mock_chat_anthropic_class.return_value = mock_instance
        mock_generate_prompt.return_value = "mocked prompt"

        result = answer_question(user_question="test question", context="test context")

        # Assert the result is correct
        assert result == "This is a test answer"

        # Verify the function called the model with correct prompt
        mock_generate_prompt.assert_called_once_with("test question", "test context")
        mock_instance.invoke.assert_called_once_with("mocked prompt")
