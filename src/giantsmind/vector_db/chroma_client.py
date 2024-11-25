from typing import Any, List, Tuple

from langchain.vectorstores import Chroma
from langchain_core.documents.base import Document

from giantsmind.vector_db.base import VectorDBClient


class ChromadbClient(VectorDBClient):
    def __init__(self, *args, **kwargs):
        self._chroma_db = Chroma(*args, **kwargs)

    def check_ids_exist(self, IDs: List[str]) -> List[bool]:
        results = self._chroma_db.get(where={"paper_id": {"$in": IDs}})
        IDs_res = [metadata["paper_id"] for metadata in results["metadatas"]]
        return [ID in IDs_res for ID in IDs]

    def similarity_search(self, query: str, **kwargs) -> List[Tuple[Document, float]]:
        return self._chroma_db.similarity_search_with_score(query, **kwargs)

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        return self._chroma_db.add_documents(documents, **kwargs)

    def __getattr__(self, name):
        return getattr(self._chroma_db, name)
