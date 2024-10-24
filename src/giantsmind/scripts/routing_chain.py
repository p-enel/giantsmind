from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from giantsmind.utils import utils
from giantsmind.agent.parsing_agent import parse_question as parse_question_agent
from giantsmind.vector_db.chroma_client import ChromadbClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from giantsmind.utils.local import get_local_data_path
from giantsmind.core.search import execute_content_search

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
    # Step 1: User Input Acquisition
    user_question = get_user_input()

    # Step 2: Initial Question Parsing and Intent Recognition
    parsed_elements = parse_question(user_question)

    # Step 3: Determine Required Operations and Process Accordingly
    metadata_results = None
    if parsed_elements.get("metadata_plain_text_search"):
        # Step 4: Generate SQL Query from Metadata Plain Text Description
        sql_query = generate_sql_query(parsed_elements["metadata_plain_text_search"])
        # Step 5: Execute SQL Query to Retrieve Metadata Results
        metadata_results = execute_sql_query(sql_query)

    content_results = None
    if parsed_elements.get("content_search"):
        # Step 6: Prepare Content Search Query
        content_query = prepare_content_search_query(parsed_elements["content_search"])
        # Step 7: Execute Content Search, Applying Metadata Filters if Available
        content_results = execute_content_search(content_query, metadata_results)

    # Step 8: Aggregate Results
    aggregated_context = aggregate_results(
        metadata_results=metadata_results,
        content_results=content_results,
        general_knowledge_required=parsed_elements.get("general_knowledge_required"),
    )

    # Step 9: Generate Final Answer Using Aggregated Context
    final_answer = generate_final_answer(user_question, aggregated_context)

    # Step 10: Deliver Answer to User
    display_answer(final_answer)


# Function Signatures Called in main()


def get_user_input():
    """Acquire the user's question or query."""
    return input("Please enter your question: ")


def parse_question(user_question):
    """
    Use an LLM to parse the user's question and identify:
    - metadata plain text search: Plain text description for metadata search, or None if not needed.
    - content search: Short sentence describing the content search, or None if not needed.
    - general knowledge required: Short sentence describing any general knowledge that may assist the answer, or None.

    Returns a dictionary with the three entries.
    """
    return parse_question_agent(user_question)


def generate_sql_query(metadata_plaintext):
    """
    Use an LLM to generate an SQL query based on the plain text metadata description.
    Returns the SQL query as a string.
    """
    pass


def execute_sql_query(sql_query):
    """
    Execute the SQL query against the metadata SQLite database.
    Returns the metadata results (e.g., a list of article IDs).
    """
    pass


def prepare_content_search_query(content_keywords):
    """
    Prepare the content search query using the extracted content keywords.
    """
    pass


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
