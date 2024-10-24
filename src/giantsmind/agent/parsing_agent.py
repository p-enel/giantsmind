from langchain_anthropic import ChatAnthropic
from giantsmind.utils import utils
from pathlib import Path

utils.set_env_vars()


def generate_prompt(user_question: str) -> str:
    prompt_path = Path(__file__).parent / "messages" / "parsing_prompt.txt"
    with open(prompt_path, "r") as file:
        prompt = file.read()
    return prompt.format(user_question=user_question)


def parse_question(user_question):
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

    model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    prompt = generate_prompt(user_question)
    response = model.invoke(prompt)

    # Handle multiple questions detected
    if "Error: Multiple content requests detected!" in response.content:
        return {"error": response.content.strip()}

    parsed_response = response.content.strip().split("\n")

    metadata_search = parsed_response[0].replace("Metadata Search: ", "").strip()
    content_search = parsed_response[1].replace("Content Search: ", "").strip()
    general_knowledge = parsed_response[2].replace("General Knowledge: ", "").strip()

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
