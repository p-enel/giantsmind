from typing import Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from giantsmind.agent import parsing_agent, sql_agent
from giantsmind.core import search
from giantsmind.utils import utils
from giantsmind.scripts.chat_with_docs import combine_docs

chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `LangChain`, `Anthropic`, or `Other`.

Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
    )
    | ChatAnthropic(model_name="claude-3-haiku-20240307")
    | StrOutputParser()
)
utils.set_env_vars()
chain.invoke({"question": "how do I call Anthropic?"})

# ------------------------------------------------------------------------------


def main():
    user_question = get_user_input()

    parsed_elements = parsing_agent.parse_question(user_question)

    metadata_results = None
    if parsed_elements.get("metadata_plain_text_search"):
        sql_query = sql_agent.get_sql_query(parsed_elements["metadata_plain_text_search"])
        metadata_results = sql_agent.execute_query(sql_query)

    content_results = None
    if parsed_elements.get("content_search"):
        paper_ids = extract_paper_ids(metadata_results)
        content_results = search.execute_content_search(parsed_elements["content_search"], paper_ids)

    aggregated_context = aggregate_results(
        parsed_elements=parsed_elements,
        metadata_results=metadata_results,
        content_results=content_results,
        general_knowledge_required=parsed_elements.get("general_knowledge_required"),
    )

    final_answer = generate_final_answer(user_question, aggregated_context)

    display_answer(final_answer)


# Function Signatures Called in main()


def get_user_input():
    """Acquire the user's question or query."""
    return input("Please enter your question: ")


def extract_paper_ids(metadata_results):
    """
    Extract paper IDs from metadata results.
    """
    if metadata_results:
        return [result["paper_id"] for result in metadata_results]
    return []


def format_metadata_results(metadata_results: List[Dict]) -> str:
    """Format the metadata results for display.

    metadata_results is a list of dictionaries containing metadata results.
    The following keys of the dictionary will be included:
        - "title"
        - "authors"
        - "publication_date"
        - "journal"
    """
    if not metadata_results:
        return "No metadata results found."
    formatted_results = []
    for result in metadata_results:
        formatted_result = (
            f"Title: {result['title']}\n"
            f"Authors: {result['authors']}\n"
            f"Publication Date: {result['publication_date']}\n"
            f"Journal: {result['journal']}\n"
        )
        formatted_results.append(formatted_result)
    return formatted_results


def aggregate_results(
    user_question: str,
    parsed_elements: Dict[str, str],
    metadata_results: List[Dict[str, str]] = None,
    content_results: List[Document] = None,
    general_knowledge_required: str = None,
) -> str:
    """
    Aggregate the metadata results, content results, and any general knowledge required.
    Prepare the aggregated context for generating the final answer.
    """
    aggregated_context = f"# The user question is:\n{user_question}\n\n"
    if metadata_results:
        metadata_str = (
            f"# This question required metadata search for: {parsed_elements['metadata_plain_text_search']}\n"
        )
        metadata_str += "Here are the results:\n" + format_metadata_results(metadata_results)
        aggregated_context += metadata_str
    if content_results:
        content_str = f"# This question required content search for: {parsed_elements['content_search']}\n"
        content_str += "Here are the results:\n" + combine_docs(content_results)
        aggregated_context += content_str
    if general_knowledge_required:
        aggregated_context += (
            f"# General knowledge required: {parsed_elements['general_knowledge_required']}\n"
        )
    return aggregated_context


def generate_final_answer(user_question, aggregated_context):
    """
    Use the LLM to generate the final answer based on the user's question and the aggregated context.
    Returns the final answer as a string.
    """
    pass


def display_answer(final_answer):
    """
    Present the final answer to the user.
    """
    pass


if __name__ == "__main__":
    main()
