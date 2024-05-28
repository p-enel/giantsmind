import asyncio
import os
import textwrap
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from llama_parse import LlamaParse


def print_response(response) -> None:
    response_txt = response["result"]
    for chunk in response_txt.split("\n"):
        if not chunk:
            print()
            continue
        print("\n".join(textwrap.wrap(chunk, 80, break_long_words=False)))


async def parse_document(file_path, llama_api_key, instruction):
    parser = LlamaParse(
        api_key=llama_api_key,
        result_type="markdown",
        parsing_instruction=instruction,
        max_timeout=5000,
    )
    return await parser.aload_data(file_path)


async def parse_all_documents_in_folder(folder_path, llama_api_key, instruction):
    folder = Path(folder_path)
    parsed_documents = []
    for file_path in folder.glob("*.pdf"):
        parsed_documents.extend(
            await parse_document(file_path, llama_api_key, instruction)
        )
    return parsed_documents


def save_parsed_document(parsed_doc, save_path):
    document_path = Path(save_path)
    with document_path.open("a") as f:
        f.write(parsed_doc.text)


def load_documents(document_path):
    loader = UnstructuredMarkdownLoader(document_path)
    return loader.load()


def split_documents(documents, chunk_size=2048, chunk_overlap=128):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def initialize_qdrant(
    docs, embeddings_model, db_path="./db", collection_name="document_embeddings"
):
    embeddings = FastEmbedEmbeddings(model_name=embeddings_model)
    return Qdrant.from_documents(
        docs,
        embeddings,
        path=db_path,
        collection_name=collection_name,
    )


def perform_similarity_search(qdrant, query):
    similar_docs = qdrant.similarity_search_with_score(query)
    for doc, score in similar_docs:
        print(f"text: {doc.page_content[:256]}\n")
        print(f"score: {score}")
        print("-" * 80)
        print()


def retrieve_documents(qdrant, query, k=5):
    retriever = qdrant.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def compress_retrieve_documents(retriever, query):
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever.invoke(query)


def create_qa_system(llm_model_name, retriever, temperature=0):
    llm = ChatGroq(temperature=temperature, model_name=llm_model_name)
    prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Answer the question and provide additional helpful information,
    based on the pieces of information, if applicable. Be succinct.

    Responses should be properly formatted to be easily read.
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose": True},
    )


async def main():

    instruction = """This is a scientific article. Please extract the text from the document and return it in markdown format."""
    folder_path = "/home/pierre/Data/giants/"
    parsed_documents = await parse_all_documents_in_folder(
        folder_path, os.getenv("LLAMA_API_KEY"), instruction
    )

    save_path = "data/parsed_documents.md"
    for parsed_doc in parsed_documents:
        save_parsed_document(parsed_doc, save_path)

    documents = load_documents(save_path)
    docs = split_documents(documents)
    print("len(docs) =", len(docs))
    print(docs[0].page_content)

    qdrant = initialize_qdrant(docs, "BAAI/bge-base-en-v1.5")

    query = "Is noise used in this work? If so, how is it used?"
    perform_similarity_search(qdrant, query)

    retriever = qdrant.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retrieve_documents(retriever, query)
    for doc in retrieved_docs:
        print(f"id: {doc.metadata['_id']}\n")
        print(f"text: {doc.page_content[:256]}\n")
        print("-" * 80)
        print()

    reranked_docs = compress_retrieve_documents(retriever, query)
    print("len(reranked_docs) =", len(reranked_docs))
    for doc in reranked_docs:
        print(f"id: {doc.metadata['_id']}\n")
        print(f"text: {doc.page_content[:256]}\n")
        print(f"score: {doc.metadata['relevance_score']}")
        print("-" * 80)
        print()

    qa_system = create_qa_system("llama3-70b-8192", retriever)
    response = qa_system.invoke("Is noise used in this work? If so, how is it used?")
    print_response(response)

    response = qa_system.invoke(
        "Explain to me what is the goal of the trained context neuron."
    )
    print_response(response)

    response = qa_system.invoke("What is mixed selectivity?")
    print_response(response)


def set_env_vars():
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().split("\n"):
            if line:
                key, value = line.split("=")
                os.environ[key] = value
    else:
        raise Exception(".env file not found.")


if __name__ == "__main__":
    set_env_vars()
    asyncio.run(main())

    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams
    from qdrant_client.http.models import PointStruct
    import numpy as np

    client = QdrantClient(
        url="https://3d544e6e-b679-46c2-b1b6-f2da6a192cb9.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key="Y0XCNbgLaf1BfthtOBi1tzJOnryDGCCIvR7YTFgdbzQ83NaSHoywtQ",
    )
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name="test",
        points=[
            PointStruct(id=2, vector=np.random.random((512)), payload={"key": "value"}),
        ],
    )

    search_result = client.search(
        collection_name="test",
        query_vector=np.random.random((512)),
        limit=1,
    )

    docs = [
        "Qdrant has Langchain integrations",
        "Qdrant also has Llama Index integrations",
    ]
    # metadata = [
    #     {"key": "Langchain-docs"},
    #     {"key": "Linkedin-docs"},
    # ]
    ids = [42, 2]

    # Use the new add method
    client.add(collection_name="test", documents=docs)

    vectors = [np.random.random((512)), np.random.random((512))]
    payload = [{"key": "value"}, {"key": "value"}]

    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    client.upload_collection(
        collection_name="test", vectors=vectors, payload=payload, ids=None
    )

    embed_model = "BAAI/bge-base-en-v1.5"
    embeddings = FastEmbedEmbeddings(model_name=embed_model)
