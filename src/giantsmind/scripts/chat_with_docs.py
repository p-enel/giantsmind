import textwrap
from typing import List, Tuple

from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.vectorstores import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents.base import Document
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import chain
from langchain_groq import ChatGroq
from parse_documents import create_client
from utils import set_env_vars

from giantsmind.core import process_results

MODELS = {"bge-small": {"model": "BAAI/bge-base-en-v1.5", "vector_size": 768}}


def print_response(response) -> None:
    response_txt = response["result"]
    for chunk in response_txt.split("\n"):
        if not chunk:
            print()
            continue
        print("\n".join(textwrap.wrap(chunk, 80, break_long_words=False)))


def perform_similarity_search(qdrant: Qdrant, query: str, **query_args) -> None:
    similar_docs = qdrant.similarity_search_with_score(query, **query_args)
    for doc, score in similar_docs:
        print(f"text: {doc.page_content[:256]}\n")
        print(f"score: {score}")
        print("-" * 80)
        print()


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
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose": True},
    )


async def main():
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

    response = qa_system.invoke("Explain to me what is the goal of the trained context neuron.")
    print_response(response)

    response = qa_system.invoke("What is mixed selectivity?")
    print_response(response)


class CombinePapersChunks(BaseCombineDocumentsChain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def acombine_docs(self, documents, **kwargs) -> Tuple[str, dict]:
        return self.combine_docs(documents, **kwargs)

    def combine_docs(self, documents: List[Document], **kwargs) -> Tuple[str, dict]:
        return process_results.combine_docs(documents, **kwargs), {}

    def invoke(
        self,
        input,
        config=None,
        **kwargs,
    ):
        output = super().invoke(input, config, **kwargs)
        return output["output_text"]


if __name__ == "__main__":
    collection = "test"
    embeddings_model = "bge-small"

    query = "What are properties of inter-day variations in brain functioning?"

    set_env_vars()
    embeddings = FastEmbedEmbeddings(model_name=MODELS[embeddings_model]["model"])
    qdrant = Qdrant(create_client(), collection, embeddings)
    # qdrant.similarity_search_with_score(query)
    retriever = qdrant.as_retriever(
        search_kwargs={"k": 5},
    )
    # retrieved_docs = retriever.invoke(query)
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    reranked_docs = compression_retriever.invoke(query)

    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

    prompt_template = """ Use the following excerpt of scientific
    articles to answer the user's question.  If you don't know the
    answer, just say that you don't know, don't try to make up an
    answer.

    <context>
    {context}
    </context>

    Question: {question}

    Answer the question by citing which paper was used to answer the
    question and provide additional helpful information, based on the
    pieces of information, if applicable. Be succinct.

    Responses should be properly formatted to be easily read.
    """

    summarization_template = """ Summarize the following excerpt or full
    scientific article. Do not include any additional information, you only
    need to summarize the text.

    <context>
    {context}
    </context>

    Summarize each section of the excerpt or full article. Be succinct.
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuffs",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose": True},
    )

    combine_papers = CombinePapersChunks()
    combined = combine_papers.invoke(results)

    my_first_chain = combine_papers | llm

    my_first_chain.invoke(results)

    response = qa.invoke(query)
    print_response(response)

    @chain
    def get_context(query: str):
        chain = compression_retriever | combine_papers
        return chain.invoke(query)

    @chain
    def basic_qa(query: str):
        context = get_context.invoke(query)
        # prompt_str = prompt.invoke({"context": context, "question": query})
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})
