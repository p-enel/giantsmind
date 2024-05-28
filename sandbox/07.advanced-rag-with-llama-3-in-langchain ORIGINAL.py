#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip -qqq install pip --progress-bar off')
get_ipython().system('pip -qqq install langchain-groq==0.1.3 --progress-bar off')
get_ipython().system('pip -qqq install langchain==0.1.17 --progress-bar off')
get_ipython().system('pip -qqq install llama-parse==0.1.3 --progress-bar off')
get_ipython().system('pip -qqq install qdrant-client==1.9.1  --progress-bar off')
get_ipython().system('pip -qqq install "unstructured[md]"==0.13.6 --progress-bar off')
get_ipython().system('pip -qqq install fastembed==0.2.7 --progress-bar off')
get_ipython().system('pip -qqq install flashrank==0.2.4 --progress-bar off')


# In[2]:


import os
import textwrap
from pathlib import Path

from google.colab import userdata
from IPython.display import Markdown
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

os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")


def print_response(response):
    response_txt = response["result"]
    for chunk in response_txt.split("\n"):
        if not chunk:
            print()
            continue
        print("\n".join(textwrap.wrap(chunk, 100, break_long_words=False)))


# In[3]:


get_ipython().system('mkdir data')
get_ipython().system('gdown 1ee-BhQiH-S9a2IkHiFbJz9eX_SfcZ5m9 -O "data/meta-earnings.pdf"')


# ## Document Parsing

# In[4]:


instruction = """The provided document is Meta First Quarter 2024 Results.
This form provides detailed financial information about the company's performance for a specific quarter.
It includes unaudited financial statements, management discussion and analysis, and other relevant disclosures required by the SEC.
It contains many tables.
Try to be precise while answering the questions"""

parser = LlamaParse(
    api_key=userdata.get("LLAMA_PARSE"),
    result_type="markdown",
    parsing_instruction=instruction,
    max_timeout=5000,
)

llama_parse_documents = await parser.aload_data("./data/meta-earnings.pdf")


# In[5]:


parsed_doc = llama_parse_documents[0]


# In[6]:


Markdown(parsed_doc.text[:4096])


# In[7]:


document_path = Path("data/parsed_document.md")
with document_path.open("a") as f:
    f.write(parsed_doc.text)


# ## Vector Embeddings

# In[8]:


loader = UnstructuredMarkdownLoader(document_path)
loaded_documents = loader.load()


# In[10]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
docs = text_splitter.split_documents(loaded_documents)
len(docs)


# In[11]:


print(docs[0].page_content)


# In[12]:


embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")


# In[13]:


qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    # location=":memory:",
    path="./db",
    collection_name="document_embeddings",
)


# In[14]:


get_ipython().run_cell_magic('time', '', 'query = "What is the most important innovation from Meta?"\nsimilar_docs = qdrant.similarity_search_with_score(query)\n')


# In[15]:


for doc, score in similar_docs:
    print(f"text: {doc.page_content[:256]}\n")
    print(f"score: {score}")
    print("-" * 80)
    print()


# In[16]:


get_ipython().run_cell_magic('time', '', 'retriever = qdrant.as_retriever(search_kwargs={"k": 5})\nretrieved_docs = retriever.invoke(query)\n')


# In[19]:


for doc in retrieved_docs:
    print(f"id: {doc.metadata['_id']}\n")
    print(f"text: {doc.page_content[:256]}\n")
    print("-" * 80)
    print()


# ## Reranking

# In[20]:


compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


# In[21]:


get_ipython().run_cell_magic('time', '', 'reranked_docs = compression_retriever.invoke(query)\nlen(reranked_docs)\n')


# In[22]:


for doc in reranked_docs:
    print(f"id: {doc.metadata['_id']}\n")
    print(f"text: {doc.page_content[:256]}\n")
    print(f"score: {doc.metadata['relevance_score']}")
    print("-" * 80)
    print()


# ## Q&A Over Document

# In[23]:


llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")


# In[24]:


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


# In[25]:


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, "verbose": True},
)


# In[26]:


get_ipython().run_cell_magic('time', '', 'response = qa.invoke("What is the most significant innovation from Meta?")\n')


# In[27]:


print_response(response)


# In[28]:


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, "verbose": False},
)


# In[29]:


get_ipython().run_cell_magic('time', '', 'response = qa.invoke("What is the revenue for 2024 and % change?")\n')


# In[30]:


Markdown(response["result"])


# In[41]:


get_ipython().run_cell_magic('time', '', 'response = qa.invoke("What is the revenue for 2023?")\n')


# In[42]:


print_response(response)


# In[33]:


get_ipython().run_cell_magic('time', '', 'response = qa.invoke(\n    "How much is the revenue minus the costs and expenses for 2024? Calculate the answer"\n)\n')


# In[34]:


print_response(response)


# In[35]:


get_ipython().run_cell_magic('time', '', 'response = qa.invoke(\n    "How much is the revenue minus the costs and expenses for 2023? Calculate the answer"\n)\n')


# In[36]:


print_response(response)


# In[37]:


get_ipython().run_cell_magic('time', '', 'response = qa.invoke("What is the expected revenue for the second quarter of 2024?")\n')


# In[38]:


Markdown(response["result"])


# In[39]:


get_ipython().run_cell_magic('time', '', 'response = qa.invoke("What is the overall outlook of Q1 2024?")\n')


# In[40]:


print_response(response)


# ## References
# 
# - [Meta Reports First Quarter 2024 Results](https://s21.q4cdn.com/399680738/files/doc_financials/2024/q1/Meta-03-31-2024-Exhibit-99-1_FINAL.pdf)
