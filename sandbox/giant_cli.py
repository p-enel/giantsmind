import os
import asyncio
import sys
from pathlib import Path

from first_RAG import (
    initialize_environment,
    parse_all_documents_in_folder,
    save_parsed_document,
    load_documents,
    split_documents,
    initialize_qdrant,
    perform_similarity_search,
    retrieve_documents,
    compress_retrieve_documents,
    create_qa_system,
    print_response,
)


def get_api_keys():
    groq_api_key = input("Enter your GROQ API Key: ")
    llama_api_key = input("Enter your LLAMA API Key: ")
    return groq_api_key, llama_api_key


async def parse_documents_in_folder(folder_path, llama_api_key):
    instruction = """This is a scientific article. Please extract the text from the document and return it in markdown format."""
    parsed_documents = await parse_all_documents_in_folder(
        folder_path, llama_api_key, instruction
    )
    save_path = "data/parsed_documents.md"
    for parsed_doc in parsed_documents:
        save_parsed_document(parsed_doc, save_path)
    print(f"Documents parsed and saved to {save_path}")


async def ask_question(retriever, question):
    response = retriever.invoke(question)
    print_response(response)


async def main_cli():
    groq_api_key, llama_api_key = get_api_keys()
    initialize_environment(groq_api_key, llama_api_key)

    qdrant = None
    retriever = None

    while True:
        user_input = (
            input(
                "\nType 'parse' to parse documents in a folder, 'ask' to ask a question, or 'quit' to exit: "
            )
            .strip()
            .lower()
        )

        if user_input == "parse":
            folder_path = input("Enter the folder path: ").strip()
            if not Path(folder_path).is_dir():
                print(f"{folder_path} is not a valid directory.")
                continue

            await parse_documents_in_folder(folder_path, llama_api_key)

            documents = load_documents("data/parsed_documents.md")
            docs = split_documents(documents)
            print(f"len(docs) = {len(docs)}")

            qdrant = initialize_qdrant(docs, "BAAI/bge-base-en-v1.5")
            retriever = qdrant.as_retriever(search_kwargs={"k": 5})
            print("Documents parsed and retriever initialized.")

        elif user_input == "ask":
            if retriever is None:
                print("Please parse documents first by typing 'parse'.")
                continue

            question = input("Enter your question: ").strip()
            await ask_question(retriever, question)

        elif user_input == "quit":
            print("Exiting the chat. Goodbye!")
            break

        else:
            print("Invalid command. Please try again.")


if __name__ == "__main__":
    asyncio.run(main_cli())
