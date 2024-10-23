from typing import List, Sequence

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document


def chunk_document(document: Document, chunk_size: int = 4096, chunk_overlap: int = 256) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents([document])


def chunk_documents(
    documents: Sequence[Document], chunk_size: int = 4096, chunk_overlap: int = 256
) -> List[List[Document]]:
    return [chunk_document(doc, chunk_size, chunk_overlap) for doc in documents]


def add_metadata_to_documents(metadata: dict, documents: List[Document]) -> List[Document]:
    for doc in documents:
        doc.metadata = metadata
    return documents
