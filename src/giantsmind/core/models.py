from dataclasses import dataclass, fields
from typing import List, Mapping, Optional, Sequence, TypedDict

from langchain_core.documents.base import Document


@dataclass
class MetadataResult:
    title: Optional[str] = None
    authors: Optional[str] = None
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    paper_id: Optional[str] = None

    def __init__(self, mapping: Mapping):
        for field in fields(self):
            setattr(self, field.name, mapping.get(field.name))

    def keys(self):
        return [field.name for field in fields(self) if getattr(self, field.name) is not None]

    def __iter__(self):
        """Make ParsedElements iterable over its field names."""
        for field in fields(self):
            yield field.name

    def __getitem__(self, key):
        """Allow access to fields via indexing."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Allow setting fields via indexing."""
        # Check that the value is either None or a string
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Value must be a string or None, not {type(value).__name__}")
        return setattr(self, key, value)

    def __mapping__(self):
        """Allow mapping over ParsedElements."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def __len__(self):
        return len(fields(self))


@dataclass
class ParsedElements(Mapping):
    metadata_search: Optional[str] = None
    content_search: Optional[str] = None
    general_knowledge: Optional[str] = None

    def __bool__(self):
        return any([self.metadata_search, self.content_search, self.general_knowledge])

    def __iter__(self):
        """Make ParsedElements iterable over its field names."""
        for field in fields(self):
            yield field.name

    def __getitem__(self, key):
        """Allow access to fields via indexing."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Allow setting fields via indexing."""
        # Check that the value is either None or a string
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Value must be a string or None, not {type(value).__name__}")
        return setattr(self, key, value)

    def __mapping__(self):
        """Allow mapping over ParsedElements."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def __len__(self):
        return len(fields(self))

    def __contains__(self, key):
        return key in self.__mapping__()


class SearchResults(TypedDict, total=False):
    metadata: List[MetadataResult]
    content: Sequence[Document]
    general: str
