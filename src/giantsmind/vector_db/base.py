from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from langchain_core.documents.base import Document


class VectorDBClient(ABC):

    @abstractmethod
    def check_ids_exist(self, IDs: List[str]) -> List[bool]: ...

    @abstractmethod
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]: ...

    @abstractmethod
    def similarity_search(self, query: str, **kwargs) -> List[Tuple[Document, float]]: ...
