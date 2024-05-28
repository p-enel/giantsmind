import os
import re
from pathlib import Path
from typing import Tuple

import fitz
import requests


def extract_metadata_from_pdf(pdf_path: str, verbose: bool = False) -> dict:
    """Open the PDF file."""
    pdf_document = fitz.open(pdf_path)
    metadata = pdf_document.metadata

    # Extract embedded metadata if available
    extracted_metadata = {
        "title": metadata.get("title", ""),
        "author": metadata.get("author", ""),
        "subject": metadata.get("subject", ""),
        "keywords": metadata.get("keywords", ""),
    }
    return extracted_metadata

    # if verbose:
    #     print("Extracted Embedded Metadata:")
    #     for key, value in extracted_metadata.items():
    #         print(f"{key.capitalize()}: {value}")
    #     print("\n" + "=" * 50 + "\n")

    # Try to detect DOI in the first few pages
    # doi = detect_doi_in_pdf(pdf_document)
    # if doi:
    #     if verbose:
    #         print(f"Detected DOI: {doi}\n")
    #     doi_metadata = fetch_metadata_from_doi(doi)
    #     extracted_metadata.update(doi_metadata)

    #     if verbose:
    #         print("Metadata Retrieved from DOI:")
    #         for key, value in doi_metadata.items():
    #             print(f"{key.capitalize()}: {value}")
    #         print("\n" + "=" * 50 + "\n")

    # return extracted_metadata


def match_doi_in_text(text: str) -> Tuple[str, str]:
    # Regular expression to match DOI patterns
    doi_pattern = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
    match = re.search(doi_pattern, text, re.IGNORECASE)

    if match:
        start_index = max(match.start() - 10, 0)
        end_index = min(match.end() + 10, len(text))
        context_text = text[start_index:end_index]
        return match.group(), context_text
    return "", ""


def find_doi_in_pdf(pdf_path: str) -> Tuple[str, str]:
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Iterate through the pages of the PDF document
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")
        # Search for DOI pattern in the extracted text
        doi, context = match_doi_in_text(text)
        if doi:
            return doi, context

    # If no DOI found in the first 5 pages
    return "", ""


def fetch_metadata_from_doi(doi: str, verbose: bool = False) -> dict:
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json().get("message", {})

        metadata = {
            "title": data.get("title", [""])[0],
            "author": ", ".join(
                [
                    author.get("family", "") + ", " + author.get("given", "")
                    for author in data.get("author", [])
                ]
            ),
            "url": data.get("URL", ""),
            "journal": data.get("container-title", [""])[0],
            "publication_date": "-".join(
                [
                    str(part)
                    for part in data.get("issued", {}).get("date-parts", [[None]])[0]
                ]
            ),
            "doi": doi,
        }

        # Format the publication date
        date_parts = metadata["publication_date"].split("-")
        if len(date_parts) > 2:
            date_parts[2] = f"{int(date_parts[2]):02d}"
        else:
            date_parts.append("01")
        date_parts[1] = f"{int(date_parts[1]):02d}"
        metadata["publication_date"] = "-".join(date_parts)

        if verbose:
            print(f"Successfully fetched metadata from DOI: {doi}")
        return metadata

    else:
        if verbose:
            print(f"Failed to fetch metadata from DOI: {doi}")
        return {}


def get_metadata(pdf_path: str | Path, verbose: bool = False) -> dict:
    # Extract metadata from the PDF file
    metadata = extract_metadata_from_pdf(pdf_path, verbose=True)

    # Try to extract DOI from the metadata
    doi, context = match_doi_in_text(metadata["subject"])
    doi_metadata = {}

    if doi:
        doi_metadata = fetch_metadata_from_doi(doi, verbose=True)

    # If DOI not found or invalid in metadata, search for it in the PDF text
    if not doi_metadata:
        doi, context = find_doi_in_pdf(pdf_path)
        if doi:
            doi_metadata = fetch_metadata_from_doi(doi, verbose=True)

    # Some PDF have issues with underscores: trying to replace spaces
    # with underscores to see if it helps
    if doi and not doi_metadata:
        new_context = context.replace(" ", "_")
        doi, context = match_doi_in_text(new_context)
        if doi:
            doi_metadata = fetch_metadata_from_doi(doi, verbose=True)

    if doi_metadata:
        metadata.update(doi_metadata)

    return metadata


def deal_with_missing_fields(metadata: dict, pdf_path: str) -> dict:
    mandatory_fields = ["title", "author", "journal", "publication_date"]
    missing_fields = [
        key
        for key in mandatory_fields
        if key not in metadata.keys() or not metadata[key]
    ]

    if not missing_fields:
        return metadata

    if "title" in missing_fields and "title" in metadata and metadata["title"]:
        paper = f'paper "{metadata["title"]}"'
    else:
        paper = f'paper with path "{pdf_path}"'
    while True:
        doi = input(
            f"Fields {missing_fields} is/are missing for {paper}.\n"
            f'If you have the DOI, please enter it (type "open" to open the file).\n'
            f"Otherwise press enter to manually ENTER them.\n"
        )
        if doi == "open":
            os.system(f'xdg-open "{pdf_path}"&')
            continue
        if not doi:
            break
        doi_metadata = fetch_metadata_from_doi(doi, verbose=True)
        if doi_metadata:
            metadata.update(doi_metadata)
            break
        else:
            print("The DOI you entered is invalid. Please try again.")
    missing_fields = [
        key
        for key in mandatory_fields
        if key not in metadata.keys() or not metadata[key]
    ]
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


if __name__ == "__main__":
    pdf_path = "/home/pierre/snap/zotero-snap/common/Zotero/storage/CVMTKL59/Shen et al. - 2022 - VILA Improving Structured Content Extraction from.pdf"
    pdf_path = "/home/pierre/snap/zotero-snap/common/Zotero/storage/DFNICQGW/10.1126@science.aar6404.pdf"
    metadata = get_metadata(pdf_path, verbose=True)
    metadata = deal_with_missing_fields(metadata, pdf_path)

    for key, value in metadata.items():
        print(f"{key.capitalize()}: {value}")
