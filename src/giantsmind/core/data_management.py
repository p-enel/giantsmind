from pathlib import Path
from typing import Dict, List

from giantsmind.metadata_db import collection_operations as col_ops
from giantsmind.utils import local


def load_markdown_paper(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()


def convert_pdf_path_to_md_fname(pdf_path: str) -> str:
    markdown_path = Path(local.get_local_data_path()) / "parsed_docs" / (Path(pdf_path).stem + ".md")
    return str(markdown_path)


def get_paper_txts_from_collection_id(collection_id: int) -> List[str]:
    paper_paths = col_ops.get_paper_paths_from_collection_id(collection_id)
    markdown_paths = [convert_pdf_path_to_md_fname(p) for p in paper_paths]
    paper_texts = [load_markdown_paper(p) for p in markdown_paths]
    return paper_texts


def combine_metadata_and_txt(metadata: Dict[str, str], paper_txt: str) -> str:
    output_txt = f"""<paper>
<title> {metadata["title"]} </title>
<authors> {metadata["authors"]} </authors>
<journal> {metadata["journal"]} </journal>
<publication date> {metadata["publication_date"]} </publication date>
<paper ID> {metadata["paper_id"]} </paper ID>
<body> {paper_txt} </body>
</paper>
"""
    return output_txt


def add_separator_to_txts(txts: List[str]) -> str:
    return "\n".join([t + "\n" + "-" * 80 for t in txts])


def get_context_from_collection(name: str) -> Dict[str, str]:
    collection_id = col_ops.get_collection_id(name)
    paper_txts = get_paper_txts_from_collection_id(collection_id)
    metadatas = col_ops.get_metadata_from_collection_id(collection_id)
    paper_contexts = [combine_metadata_and_txt(m, t) for m, t in zip(metadatas, paper_txts)]
    context = add_separator_to_txts(paper_contexts)
    return context
