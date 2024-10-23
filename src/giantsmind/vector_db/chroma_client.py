from typing import List

from langchain.vectorstores import Chroma

from giantsmind.vector_db.base import VectorDBClient


class ChromadbClient(Chroma, VectorDBClient):

    def check_ids_exist(self, IDs: List[str]) -> List[bool]:
        results = self.get(where={"paper_id": {"$in": IDs}})
        IDs_res = [metadata["paper_id"] for metadata in results["metadatas"]]
        exists = [ID in IDs_res for ID in IDs]
        return exists
