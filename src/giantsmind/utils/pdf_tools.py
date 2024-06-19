import hashlib
from pathlib import Path
from typing import List

from PyPDF2 import PdfReader, PdfWriter


def get_pdf_hashes(files: List[str]) -> List[str]:
    """Get the hash of each file in a list of files."""
    hashes = []
    for file in files:
        with open(file, "rb") as f:
            data = f.read()
            hash = hashlib.sha256(data).hexdigest()
            hashes.append(hash)
    return hashes


def split_pdf(input_pdf: str, output_folder: str) -> None:
    input_path = Path(input_pdf)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_reader = PdfReader(input_pdf)
    base_filename = input_path.stem

    for page_num in range(len(pdf_reader.pages)):
        pdf_writer = PdfWriter()
        pdf_writer.add_page(pdf_reader.pages[page_num])

        output_filename = output_path / f"{base_filename}_page_{page_num + 1}.pdf"
        with open(output_filename, "wb") as output_pdf:
            pdf_writer.write(output_pdf)
