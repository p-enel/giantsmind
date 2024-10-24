from pathlib import Path
from typing import Dict, Optional

from langchain_anthropic import ChatAnthropic
from giantsmind.utils import utils

utils.set_env_vars()

PROMPT_PATH = Path(__file__).parent / "messages" / "answering_prompt.txt"


def generate_answering_prompt(user_question: str, context: str) -> str:
    with open(PROMPT_PATH, "r") as file:
        prompt = file.read()
    return prompt.format(user_question=user_question, context=context)


def answer_question(user_question: str, context: str) -> str:
    model = ChatAnthropic(model="claude-3-5-sonnet-latest")
    prompt = generate_answering_prompt(context)
    response = model.invoke(prompt)
    return response.content.strip()


def invoke(user_question: str, context: str) -> str:
    return answer_question(user_question, context)


# Example usage
if __name__ == "__main__":
    user_question = "What are the key findings of the paper on deep learning?"
    context = "The paper discusses various deep learning techniques and their applications."
    answer = invoke(user_question, context)
    print(answer)
