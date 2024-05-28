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
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from llama_parse import LlamaParse


GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
LLAMA_API_KEY = "llx-kVRDSeO1yUpeSqEWuKvpJ9utAxTsdOeEPsEAXLBRG59xteSq"


def print_response(response):
    response_txt = response["result"]
    for chunk in response_txt.split("\n"):
        if not chunk:
            print()
            continue
        print("\n".join(textwrap.wrap(chunk, 100, break_long_words=False)))


instruction = """This is a scientific article. Please extract the text from the document and return it in markdown format."""

parser = LlamaParse(
    api_key=LLAMA_API_KEY,
    result_type="markdown",
    parsing_instruction=instruction,
    max_timeout=5000,
)

llama_parse_documents = await parser.aload_data(
    "/home/pierre/Data/giants/1-s2.0-S1053811922000088-main.pdf"
)

parsed_doc = llama_parse_documents[0]


document_path = Path("data/parsed_document.md")
with document_path.open("w") as f:
    f.write(parsed_doc.text)


loader = UnstructuredMarkdownLoader(document_path)
loaded_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
docs = text_splitter.split_documents(loaded_documents)
print("len(docs) =", len(docs))


print(docs[0].page_content)

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    # location=":memory:",
    path="./db",
    collection_name="document_embeddings",
)

query = "How many microstate were found in this study?"
similar_docs = qdrant.similarity_search_with_score(query)


for doc, score in similar_docs:
    print(f"text: {doc.page_content[:256]}\n")
    print(f"score: {score}")
    print("-" * 80)
    print()

retriever = qdrant.as_retriever(search_kwargs={"k": 5})
retrieved_docs = retriever.invoke(query)


for doc in retrieved_docs:
    print(f"id: {doc.metadata['_id']}\n")
    print(f"text: {doc.page_content[:256]}\n")
    print("-" * 80)
    print()


compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

reranked_docs = compression_retriever.invoke(query)
print("len(reranked_docs) =", len(reranked_docs))


for doc in reranked_docs:
    print(f"id: {doc.metadata['_id']}\n")
    print(f"text: {doc.page_content[:256]}\n")
    print(f"score: {doc.metadata['relevance_score']}")
    print("-" * 80)
    print()


llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

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


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, "verbose": True},
)


response = qa.invoke("What microstates found in this study? How many?")
print_response(response)


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, "verbose": False},
)

response = qa.invoke("Explain to me what is the goal of the trained context neuron.")
print_response(response)

response = qa.invoke("What is mixed selectivity?")
print_response(response)
