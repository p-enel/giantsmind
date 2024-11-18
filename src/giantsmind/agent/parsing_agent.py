from importlib import resources
from typing import Dict, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

PROMPT_PATH = resources.files("giantsmind.agent.resources.messages") / "parsing_prompt.txt"
ERROR_MESSAGE = "Error:"  # Multiple content requests detected!"


def generate_prompt(user_question: str) -> str:
    with open(PROMPT_PATH, "r") as file:
        prompt = file.read()
    return prompt.format(user_question=user_question)


def parse_question(user_question: str) -> Dict[str, Optional[str]]:
    """
    Parses the user's question into three parts:
    - Metadata plain text search
    - Content search
    - General knowledge required

    Args:
        user_question (str): The question provided by the user.

    Returns:
        dict: A dictionary containing the parsed outputs: metadata_search, content_search, general_knowledge.
    """

    model = ChatAnthropic(model="claude-3-5-sonnet-latest")
    prompt = generate_prompt(user_question)
    response = model.invoke(prompt)

    # Handle multiple questions detected
    if ERROR_MESSAGE in response.content:
        return {"error": response.content.strip()}

    parsed_response = response.content.strip().split("\n")

    metadata_search = parsed_response[0].replace("Metadata Search: ", "").strip()
    content_search = parsed_response[1].replace("Content Search: ", "").strip()
    general_knowledge = parsed_response[2].replace("General Knowledge: ", "").strip()

    # Make sure the strings are replaced with None if the string is "None"
    metadata_search = None if metadata_search.lower().startswith("none") else metadata_search
    content_search = None if content_search.lower().startswith("none") else content_search
    general_knowledge = None if general_knowledge.lower().startswith("none") else general_knowledge

    return {
        "metadata_search": metadata_search,
        "content_search": content_search,
        "general_knowledge": general_knowledge,
    }


# Example usage
if __name__ == "__main__":
    user_question = (
        "Find all papers published by Pierre Enel or Peter Dominey on reservoir computing since 2010."
    )
    parsed_question = parse_question(user_question)
    if "error" in parsed_question:
        print(parsed_question["error"])
    else:
        print(parsed_question)
