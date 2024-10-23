from typing import List
from abc import ABC, abstractmethod
from langchain_core.documents.base import Document


class VectorDBClient(ABC):

    @abstractmethod
    def check_ids_exist(self, IDs: List[str]) -> List[bool]:
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        pass
