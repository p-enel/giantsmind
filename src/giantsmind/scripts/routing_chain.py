from giantsmind.agent import answering_agent, sql_agent
from giantsmind.core import process_results as proc_res
from giantsmind.core import search
from giantsmind.utils.logging import logger


def main():
    # user_question = input("Please enter your question: ")
    user_question = "How is a microstate defined in the papers in the database?"

    # parsed_elements = parsing_agent.parse_question(user_question)

    parsed_elements = {
        "metadata_search": 'Retrieve papers containing "microstate" in their metadata',
        "content_search": "definition of microstate",
        "general_knowledge": "Provide general context on microstates in physics/thermodynamics if needed",
    }

    if "error" in parsed_elements:
        print(parsed_elements["error"])
        return

    metadata_results = None
    if parsed_elements.get("metadata_search"):
        # sql_query = sql_agent.get_sql_query(parsed_elements["metadata_search"])
        sql_query = "SQL: SELECT DISTINCT p.paper_id \nFROM papers p\nJOIN paper_collection pc ON p.paper_id = pc.paper_id\nWHERE p.title LIKE '%microstate%'\nAND pc.collection_id = 1;"
        metadata_results = sql_agent.execute_metadata_query(sql_query)

    content_results = None
    if parsed_elements.get("content_search"):
        paper_ids = proc_res.extract_paper_ids(metadata_results)
        content_results, content_scores = search.execute_content_search(
            parsed_elements["content_search"], paper_ids
        )

    aggregated_context = proc_res.aggregate_results(
        user_question,
        parsed_elements,
        metadata_results=metadata_results,
        content_results=content_results,
        general_knowledge=parsed_elements.get("general_knowledge"),
    )

    final_answer = answering_agent.invoke(user_question, aggregated_context)

    print(f"\nAnswer: {final_answer}")


if __name__ == "__main__":
    main()
    # What is the subject of articles published after 2020?
    # Give me a list of the 10 first papers in the database.
