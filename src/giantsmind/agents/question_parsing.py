from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Callable, Dict, List, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Load environment variables
load_dotenv()


@dataclass
class ParserConfig:
    """Configuration class for the parser"""

    prompt_path: Path
    model_name: str
    error_message: str = "Error:"
    search_prefixes: Dict[str, str] = None

    def __post_init__(self):
        if self.search_prefixes is None:
            self.search_prefixes = {
                "metadata_search": "Metadata Search: ",
                "content_search": "Content Search: ",
                "general_knowledge": "General Knowledge: ",
            }


class ModelClient(ABC):
    """Abstract base class for AI model clients"""

    @abstractmethod
    def get_response(self, prompt: str) -> str:
        """Get response from the AI model"""
        pass


class AnthropicClient(ModelClient):
    """Implementation of ModelClient for Anthropic's Claude"""

    def __init__(self, model_name: str):
        self.model = ChatAnthropic(model=model_name)

    def get_response(self, prompt: str) -> str:
        """Get response from Claude"""
        return self.model.invoke(prompt).content.strip()


def generate_prompt(user_question: str, prompt_path: Path) -> str:
    """Generate prompt from template file"""
    with open(prompt_path, "r") as file:
        prompt_template = file.read()
    return prompt_template.format(user_question=user_question)


class ResponseParser:
    """Class to handle parsing of model responses"""

    def __init__(self, config: ParserConfig):
        self.config = config

    def validate_response(self, response: str) -> Optional[Dict[str, str]]:
        """Validate the model's response"""
        if self.config.error_message in response:
            return {"error": response.strip()}
        return None

    def split_response(self, response: str) -> List[str]:
        """Split response into lines"""
        return [line.strip() for line in response.strip().split("\n")]

    def extract_search_value(self, line: str, prefix: str) -> Optional[str]:
        """Extract search value from a line given a prefix"""
        value = line.replace(prefix, "").strip()
        return None if value.lower().startswith("none") else value

    def extract_search_components(self, lines: List[str]) -> Dict[str, Optional[str]]:
        """Extract search components from response lines"""
        return {
            search_type: self.extract_search_value(lines[i], prefix)
            for i, (search_type, prefix) in enumerate(self.config.search_prefixes.items())
        }


class QuestionParser:
    """Main class for parsing questions"""

    def __init__(
        self,
        config: ParserConfig,
        model_client: ModelClient,
        generate_prompt: Callable[[str], str],
        response_parser: ResponseParser,
    ):
        self.config = config
        self.model_client = model_client
        self.generate_prompt = generate_prompt
        self.response_parser = response_parser

    def parse_question(self, user_question: str) -> Dict[str, Optional[str]]:
        """Parse user question into search components"""
        prompt = self.generate_prompt(user_question, self.config.prompt_path)
        response = self.model_client.get_response(prompt)

        error_result = self.response_parser.validate_response(response)
        if error_result:
            return error_result

        lines = self.response_parser.split_response(response)
        return self.response_parser.extract_search_components(lines)


def create_default_parser() -> QuestionParser:
    """Factory function to create a QuestionParser with default configuration"""
    config = ParserConfig(
        prompt_path=resources.files("giantsmind.agents.resources.messages") / "parsing_prompt.txt",
        model_name="claude-3-5-sonnet-latest",
    )
    model_client = AnthropicClient(config.model_name)
    response_parser = ResponseParser(config)
    return QuestionParser(config, model_client, generate_prompt, response_parser)


# Example usage
if __name__ == "__main__":
    # Create parser with default configuration
    parser = create_default_parser()

    # Example question
    user_question = (
        "Find all papers published by Pierre Enel or Peter Dominey on reservoir computing since 2010."
    )

    # Parse question
    parsed_question = parser.parse_question(user_question)

    # Handle result
    if "error" in parsed_question:
        print(parsed_question["error"])
    else:
        print(parsed_question)
