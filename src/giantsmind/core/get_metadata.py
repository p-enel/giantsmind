import os
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import fitz
import pandas as pd
import PyPDF2
import requests

from giantsmind.utils import local
from giantsmind.database.schema import Session
from giantsmind.database.operations import add_paper


def extract_metadata_from_pdf(pdf_path: str, verbose: bool = False) -> dict:
    """Open the PDF file."""
    pdf_document = fitz.open(pdf_path)
    metadata = pdf_document.metadata

    # Extract embedded metadata if available
    extracted_metadata = {
        "title": metadata.get("title", ""),
        "author": metadata.get("author", ""),
        "subject": metadata.get("subject", ""),
    }
    return extracted_metadata


def match_doi_in_text(text: str, context_length: int = 10) -> Tuple[str, str]:
    # Regular expression to match DOI patterns
    doi_pattern = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
    match = re.search(doi_pattern, text, re.IGNORECASE)

    if match:
        start_index = max(match.start() - context_length, 0)
        end_index = min(match.end() + context_length, len(text))
        context_text = text[start_index:end_index]
        return match.group(), context_text
    return "", ""


def find_doi_in_pdf(pdf_path: str, n_first_pages: int = 5, context_length: int = 10) -> Tuple[str, str]:
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Iterate through the pages of the PDF document
    for page_num in range(min(len(pdf_document), 5)):
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")
        # Search for DOI pattern in the extracted text
        doi, context = match_doi_in_text(text)
        if doi:
            doi.rstrip(".")
            return doi, context

    # If no DOI found in the first 5 pages
    return "", ""


def find_arxiv_id_in_text(text: str, context_length: int = 5) -> Tuple[str, str]:
    # Regular expression pattern for arXiv ID
    arxiv_pattern = r"arXiv:\d{4}\.\d{4,5}(v\d+)?"

    # Search for the arXiv ID pattern in the text
    match = re.search(arxiv_pattern, text)

    if match:
        start = max(0, match.start() - context_length)
        end = min(len(text), match.end() + context_length)
        context = text[start:end]
        return match.group(0).split(":")[1], context

    return "", ""


def find_arxiv_id_in_pdf(pdf_path: str) -> Tuple[str, str]:
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Iterate through the pages of the PDF document, up to 5 pages
    for page_num in range(min(5, len(pdf_document))):
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")
        # Search for arXiv ID pattern in the extracted text
        arxiv_id, context = find_arxiv_id_in_text(text)
        if arxiv_id:
            return arxiv_id, context

    # If no arXiv ID found in the first 5 pages
    return "", ""


def fetch_metadata_from_doi(doi: str, verbose: bool = False) -> dict:
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)

    if response.status_code != 200:
        if verbose:
            print(f"Failed to fetch metadata from DOI: {doi}")
        return {}

    data = response.json().get("message", {})

    title = data.get("title")[0]
    authors = "; ".join(
        f"{author.get('given', '')} {author.get('family', '')}" for author in data.get("author", [])
    )
    url = data.get("URL", "")

    journal = data.get("container-title", [None])
    if isinstance(journal, list) and len(journal) > 0:
        journal = journal[0]
    if not journal and data.get("short-container-title"):
        journal = data.get("short-container-title")[0]
    if not journal and data.get("institution"):
        journal = data.get("institution")[0]["name"]

    date_parts = data.get("published", {}).get("date-parts", [None])[0]
    if not date_parts or len(date_parts) != 3:
        date_parts = data.get("published-online", {}).get("date-parts", [date_parts])[0]
    if not date_parts and journal == "bioRxiv":
        date_parts = data.get("posted", {}).get("date-parts", [None])[0]
    if date_parts is None and verbose:
        print("Publication date not found in the DOI data.")

    # Ensure date format is YYYY-MM-DD
    while len(date_parts) < 3:
        date_parts.append(1)
    year, month, day = date_parts[:3]
    publication_date = f"{year:04d}-{month:02d}-{day:02d}"

    metadata = {
        "title": title,
        "author": authors,
        "url": url,
        "journal": journal,
        "publication_date": publication_date,
        "id": f"doi:{doi}",
    }

    if verbose:
        print(f"Successfully fetched metadata from DOI: {doi}")

    return metadata


def fetch_metadata_from_arxiv(arxiv_id: str, verbose: bool = False) -> dict:
    base_url = "http://export.arxiv.org/api/query"
    params = {"id_list": arxiv_id}
    response = requests.get(base_url, params=params)

    if response.status_code != 200:
        if verbose:
            print(f"Error: {response.status_code}")
        return {}

    root = ET.fromstring(response.content)
    entry = root.find("{http://www.w3.org/2005/Atom}entry")

    if entry is None:
        if verbose:
            print("No entry found for the given arXiv ID.")
        return {}

    title = entry.find("{http://www.w3.org/2005/Atom}title").text.replace("\n", " ")
    title = " ".join(title.split())
    authors = "; ".join(
        author.find("{http://www.w3.org/2005/Atom}name").text
        for author in entry.findall("{http://www.w3.org/2005/Atom}author")
    )
    url = entry.find("{http://www.w3.org/2005/Atom}id").text
    published = entry.find("{http://www.w3.org/2005/Atom}published").text
    journal = "arXiv"  # arXiv papers typically don't have a journal, so we'll set it to 'arXiv'

    # Format the publication date to YYYY-MM-DD
    publication_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")

    metadata = {
        "title": title,
        "author": authors,
        "url": url,
        "journal": journal,
        "publication_date": publication_date,
        "id": f"arXiv:{arxiv_id}",
    }

    if verbose:
        print(f"Successfully fetched metadata from arXiv ID: {arxiv_id}")

    return metadata


def get_doi_metadata(metadata: dict, pdf_path: str, verbose: bool) -> dict:
    doi, context = match_doi_in_text(metadata.get("subject", ""))
    if doi:
        doi_metadata = fetch_metadata_from_doi(doi, verbose=verbose)
        if doi_metadata:
            return doi_metadata

    doi, context = find_doi_in_pdf(pdf_path)
    if doi:
        doi_metadata = fetch_metadata_from_doi(doi, verbose=verbose)
        if doi_metadata:
            return doi_metadata

    if doi:
        new_context = context.replace(" ", "_")
        doi, _ = match_doi_in_text(new_context)
        if doi:
            doi_metadata = fetch_metadata_from_doi(doi, verbose=verbose)
            if doi_metadata:
                return doi_metadata

    return {}


def get_arxiv_metadata(metadata: dict, pdf_path: str, verbose: bool) -> dict:
    arxiv_id, context = find_arxiv_id_in_text(metadata.get("subject", ""))
    if arxiv_id:
        arxiv_metadata = fetch_metadata_from_arxiv(arxiv_id, verbose=verbose)
        if arxiv_metadata:
            return arxiv_metadata

    arxiv_id, context = find_arxiv_id_in_pdf(pdf_path)
    if arxiv_id:
        arxiv_metadata = fetch_metadata_from_arxiv(arxiv_id, verbose=verbose)
        if arxiv_metadata:
            return arxiv_metadata

    return {}


def get_metadata(pdf_path: str, verbose: bool = False) -> dict:
    # Extract metadata from the PDF file
    if verbose:
        print(f"Extracting metadata from PDF: {pdf_path}")
    metadata_pdf = extract_metadata_from_pdf(pdf_path, verbose=verbose)

    # Try to fetch DOI metadata
    doi_metadata = get_doi_metadata(metadata_pdf, pdf_path, verbose)
    if doi_metadata:
        if verbose:
            print("    Extracted metadata from DOI.")
        return doi_metadata

    # Try to fetch arXiv metadata
    arxiv_metadata = get_arxiv_metadata(metadata_pdf, pdf_path, verbose)
    if arxiv_metadata:
        if verbose:
            print("    Extracted metadata from arXiv ID.")
        return arxiv_metadata

    if verbose:
        print("    No metadata found from DOI or arXiv ID")

    # Remove extra fields from the metadata
    all_fields = ["title", "author", "journal", "publication_date", "ID", "url"]
    metadata_pdf = {key: metadata_pdf.get(key, "") for key in metadata_pdf if key in all_fields}
    return metadata_pdf


def deal_with_missing_fields(metadata: dict, pdf_path: str) -> dict:
    mandatory_fields = ["title", "author", "journal", "publication_date"]
    missing_fields = [key for key in mandatory_fields if key not in metadata.keys() or not metadata[key]]

    if not missing_fields:
        return metadata

    if "title" in missing_fields and "title" in metadata and metadata["title"]:
        paper = f'paper "{metadata["title"]}"'
    else:
        paper = f'paper with path "{pdf_path}"'
    while True:
        choice = input(
            "------------------------------------------------------------\n"
            f"Fields {missing_fields} is/are missing for {paper}.\n"
            f'If you have the DOI, press "d + ENTER", if you have the arXiv ID, type "a + ENTER" or type "o + ENTER" to open the PDF file).\n'
            f"Otherwise press ENTER to manually enter these data.\n"
        )
        if choice == "open":
            os.system(f'xdg-open "{pdf_path}"&')
            continue
        if choice == "d":
            doi = input("Please enter the DOI: ")
            doi_metadata = fetch_metadata_from_doi(doi, verbose=True)
            if doi_metadata:
                metadata.update(doi_metadata)
                ask_to_edit_metadata_pdf(pdf_path, metadata)
                break
            else:
                print("The DOI you entered is likely invalid.\n")
            continue
        if choice == "a":
            arxiv_id = input("Please enter the arXiv ID: ")
            arxiv_metadata = fetch_metadata_from_arxiv(arxiv_id, verbose=True)
            if arxiv_metadata:
                metadata.update(arxiv_metadata)
                ask_to_edit_metadata_pdf(pdf_path, metadata)
                break
            else:
                print("The arXiv ID you entered is likely invalid.\n")
            continue
        if not choice:
            break
        doi_metadata = fetch_metadata_from_doi(doi, verbose=True)
        if doi_metadata:
            metadata.update(doi_metadata)
            break
        else:
            print("Invalid choice. Please try again.\n")

    missing_fields = [key for key in mandatory_fields if key not in metadata.keys() or not metadata[key]]
    if not missing_fields:
        return metadata
    print("Please enter the missing fields manually. Type 'open' to open the file.")
    for field in missing_fields:
        value = input(f"Please enter the {field}: ")
        while value == "open":
            os.system(f'xdg-open "{pdf_path}"&')
            value = input(f"Please enter the {field}: ")
        metadata[field] = value
    return metadata


def edit_pdf_metadata(pdf_path: str, output_path: str, new_metadata: dict):
    # Open the existing PDF
    with open(pdf_path, "rb") as input_pdf:
        reader = PyPDF2.PdfReader(input_pdf)
        writer = PyPDF2.PdfWriter()

        # Add all pages to the writer
        for page_num in range(len(reader.pages)):
            writer.add_page(reader.pages[page_num])

        # Update the metadata
        writer.add_metadata(new_metadata)

        # Write out the updated PDF
        with open(output_path, "wb") as output_pdf:
            writer.write(output_pdf)


def ask_to_edit_metadata_pdf(pdf_path: str, metadata: dict):
    print("Here is the metadata we have on this paper:")
    for key, value in metadata.items():
        print(f"- {key}: {value}")

    choice = input("Would you like to edit the metadata? ([y]/n): ")
    if not (choice.lower() == "y" or not choice):
        return

    subject = f"{metadata['journal']} ({metadata['publication_date']}) ID: {metadata['id']}"
    metadata_for_df = {
        "/Title": metadata["title"],
        "/Author": metadata["author"],
        "/Subject": subject,
    }
    edit_pdf_metadata(pdf_path, pdf_path, metadata_for_df)
    print(f"Metadata for {pdf_path} has been updated.")


def fetch_and_process_metadata(files: Sequence[str], verbose: bool) -> List[dict]:
    metadatas = [get_metadata(file, verbose=verbose) for file in files]
    return [deal_with_missing_fields(metadata, pdf_path) for metadata, pdf_path in zip(metadatas, files)]


def add_file_path_to_metadata(metadatas: List[dict], pdf_paths: List[str]) -> List[dict]:
    metadatas = deepcopy(metadatas)
    new_metadatas = []
    for metadata, file_path in zip(metadatas, pdf_paths):
        metadata["file_path"] = file_path
        new_metadatas.append(metadata)
    return new_metadatas


def update_metadata_df(metadatas: List[dict]) -> None:
    """Update the metadata dataframe with new metadata."""
    session = Session()
    for metadata in metadatas:
        add_paper(session, metadata)


def process_metadata(files: Sequence[str], hashes: Sequence[str], verbose: bool = True) -> List[dict]:
    """Get and save metadata for a list of files."""
    if len(files) != len(hashes):
        raise ValueError("Number of files and hashes do not match.")
    if not files and verbose:
        print("No files to process.")
        return

    metadatas = fetch_and_process_metadata(files, verbose)
    update_metadata_df(metadatas, hashes, files)
    return metadatas
