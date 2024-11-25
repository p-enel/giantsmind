from typing import List, Sequence, TypedDict

from langchain_core.documents.base import Document


class MetadataResult(TypedDict):
    title: str
    authors: str
    publication_date: str
    journal: str
    paper_id: str


class ParsedElements(TypedDict, total=False):
    metadata_search: str
    content_search: str
    general_knowledge: str


class SearchResults(TypedDict, total=False):
    metadata: List[MetadataResult]
    content: Sequence[Document]
    general: str
