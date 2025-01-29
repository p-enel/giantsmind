from importlib import resources
from typing import Callable

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

PROMPT_PATH = resources.files("giantsmind.agents.resources.messages").joinpath("answering_prompt.txt")


def generate_answering_prompt(user_question: str, context: str) -> str:
    with open(PROMPT_PATH, "r") as file:
        prompt = file.read()
    return prompt.format(user_question=user_question, context=context)


def answer_question(
    user_question: str, context: str, prompt_generator: Callable[[str, str], str] = generate_answering_prompt
) -> str:
    model = ChatAnthropic(model="claude-3-5-sonnet-latest")
    prompt = prompt_generator(user_question, context)
    response = model.invoke(prompt)
    return response.content.strip()


def invoke(
    user_question: str, context: str, answer_question_func: Callable[[str, str], str] = answer_question
) -> str:
    return answer_question_func(user_question, context)


# Example usage
if __name__ == "__main__":
    user_question = "What are the key findings of the paper on deep learning?"
    context = "The paper discusses various deep learning techniques and their applications."
    answer = invoke(user_question, context)
    print(answer)
