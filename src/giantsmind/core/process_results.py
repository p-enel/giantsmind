from typing import List, Dict

from langchain_core.documents.base import Document


def extract_paper_ids(metadata_results):
    """
    Extract paper IDs from metadata results.
    """
    if metadata_results:
        return [result["paper_id"] for result in metadata_results]
    return []


def combine_docs(documents: List[Document], **kwargs) -> str:
    output = ["-" * 80]
    for doc in documents:
        metadata_str = f"The following text is an excerpt of the article entitled '{doc.metadata['title']}' from author(s) {doc.metadata['authors']}. It was published in {doc.metadata['journal']} in (year-month-day) {doc.metadata['publication_date']}. Paper ID: {doc.metadata['paper_id']}.\n\n<excerpt>"
        output.append(metadata_str)
        output.append(doc.page_content)
        output.append("</excerpt>\n" + "-" * 80)
    output.append("End of excerpts.")
    return "\n".join(output)


def format_metadata_results(metadata_results: List[Dict]) -> str:
    """Format the metadata results for display.

    metadata_results is a list of dictionaries containing metadata results.
    The following keys of the dictionary will be included:
        - "title"
        - "authors"
        - "publication_date"
        - "journal"
        - "paper_id"
    """
    if not metadata_results:
        return "No metadata results found."
    formatted_txt = ""
    for result in metadata_results:
        formatted_result = f"""Title: {result['title']}
Authors: {result['authors']}
Publication Date: {result['publication_date']}
Journal: {result['journal']}
Paper ID: {result['paper_id']}
"""
        formatted_txt += formatted_result
    return formatted_txt


def aggregate_results(
    user_question: str,
    parsed_elements: Dict[str, str],
    metadata_results: List[Dict[str, str]] = None,
    content_results: List[Document] = None,
    general_knowledge: str = None,
) -> str:
    """
    Aggregate the metadata results, content results, and any general knowledge required.
    Prepare the aggregated context for generating the final answer.
    """
    aggregated_context = ""
    if metadata_results:
        metadata_str = (
            f"# This question required a metadata search: \"{parsed_elements['metadata_search']}\"\n"
        )
        metadata_str += "Here are the results:\n" + format_metadata_results(metadata_results)
        aggregated_context += metadata_str
    if content_results:
        content_str = f"# This question required content search for: {parsed_elements['content_search']}\n"
        content_str += "Here are the results:\n" + combine_docs(content_results)
        aggregated_context += content_str + "\n\n"
    if general_knowledge:
        aggregated_context += f"# General knowledge required: {parsed_elements['general_knowledge']}\n"
    return aggregated_context
