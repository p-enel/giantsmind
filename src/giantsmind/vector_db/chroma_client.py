from typing import List, Tuple

from langchain.vectorstores import Chroma
from langchain_core.documents.base import Document

from giantsmind.vector_db.base import VectorDBClient


class ChromadbClient(Chroma, VectorDBClient):

    def check_ids_exist(self, IDs: List[str]) -> List[bool]:
        results = self.get(where={"paper_id": {"$in": IDs}})
        IDs_res = [metadata["paper_id"] for metadata in results["metadatas"]]
        exists = [ID in IDs_res for ID in IDs]
        return exists

    def similarity_search(self, query: str, **kwargs) -> List[Tuple[Document, float]]:
        return Chroma.similarity_search_with_score(self, query, **kwargs)
