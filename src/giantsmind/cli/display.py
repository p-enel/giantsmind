from typing import Dict, List

from langchain_core.documents.base import Document

from giantsmind.utils import local


def author_to_str(author: str) -> str:
    authors = author.split("; ")
    if len(authors) > 1:
        return f"{authors[0]} et al."
    return authors[0]


def print_doc_list(documents: List[Document]):
    titles = [doc.metadata["paper_metadata"]["title"] for doc in documents]
    authors = [doc.metadata["paper_metadata"]["author"] for doc in documents]
    publications = [doc.metadata["paper_metadata"]["journal"] for doc in documents]
    years = [doc.metadata["paper_metadata"]["publication_date"] for doc in documents]
    ids = [doc.metadata["paper_metadata"]["ID"] for doc in documents]
    printed_ids = []
    for (
        id,
        author,
        journal,
        year,
        title,
    ) in zip(ids, authors, publications, years, titles):
        if id in printed_ids:
            continue
        author = author_to_str(author)
        print(f"{author:.20} | {journal:.15} ({year.split('-')[0]}) | {title:.49}")
        printed_ids.append(id)


def format_string(s: str, length: int) -> str:
    if len(s) > length:
        return s[: length - 1] + "*"
    else:
        return s.ljust(length)


def paper_metadata_to_str(paper_metadata: dict, element_sizes: Dict[str, int]) -> str:
    elements = {
        "author": author_to_str(paper_metadata["author"]),
        "journal": paper_metadata["journal"],
        "year": paper_metadata["publication_date"].split("-")[0],
        "title": paper_metadata["title"],
    }
    resized = {key: format_string(elements[key], element_sizes[key]) for key in elements}
    line_str = f"{resized['author']} | {resized['journal']} ({resized['year']}) | {resized['title']}"
    return line_str


def get_paper_list_header(element_sizes: Dict[str, int]) -> str:
    elements = {
        "author": "Author",
        "journal": "Journal",
        "year": "Year",
        "title": "Title",
    }
    resized = {key: format_string(elements[key], element_sizes[key]) for key in elements}
    line_str = f"{resized['author']} | {resized['journal']} ({resized['year']}) | {resized['title']}"
    return line_str


def print_numbered_list(header_str: str, list_strs: List[str], number_offset: int) -> None:
    print(" " * (number_offset + 2) + header_str)
    print("-" * (number_offset + 2 + len(header_str)))
    for i, line_str in enumerate(list_strs):
        number_str = str(i + 1).rjust(number_offset)
        print(f"{number_str}. {line_str}")


def print_id_list(id_list: List[str]) -> None:
    sizes = {"author": 20, "journal": 15, "year": 4, "title": 49}
    metadata_df = local.load_metadata_df()
    metadatas = [metadata_df.loc[id].to_dict() for id in id_list]
    paper_strs = [paper_metadata_to_str(metadata, sizes) for metadata in metadatas]
    number_offset = len(str(len(paper_strs)))
    print_numbered_list(get_paper_list_header(sizes), paper_strs, number_offset)
