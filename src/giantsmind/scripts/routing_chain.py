from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from giantsmind.utils import utils
from giantsmind.agent import parsing_agent
from giantsmind.agent import sql_agent
from giantsmind.core import search

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


def aggregate_results(metadata_results=None, content_results=None, general_knowledge_required=None):
    """
    Aggregate the metadata results, content results, and any general knowledge required.
    Prepare the aggregated context for generating the final answer.
    """
    pass


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
