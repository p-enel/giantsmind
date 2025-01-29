import warnings
from pathlib import Path
from typing import Callable, Dict, List

from giantsmind.metadata_db import collection_operations as col_ops
from giantsmind.utils import local


class FileHandler:
    def __init__(self, get_local_data_path: Callable[[], str] = local.get_local_data_path):
        self.get_local_data_path = get_local_data_path

    def load_markdown_paper(self, file_path: str) -> str:
        return open(file_path, "r").read()

    def convert_pdf_path_to_md_fname(self, pdf_path: str) -> str:
        markdown_path = Path(self.get_local_data_path()) / "parsed_docs" / (Path(pdf_path).stem + ".md")
        return str(markdown_path)


class PaperFormatter:
    @staticmethod
    def combine_metadata_and_txt(metadata: Dict[str, str], paper_txt: str) -> str:
        return f"""<paper>
<title> {metadata["title"]} </title>
<authors> {metadata["authors"]} </authors>
<journal> {metadata["journal"]} </journal>
<publication date> {metadata["publication_date"]} </publication date>
<paper ID> {metadata["paper_id"]} </paper ID>
<body> {paper_txt} </body>
</paper>
"""

    @staticmethod
    def add_separator_to_txts(txts: List[str]) -> str:
        return "\n".join([t + "\n" + "-" * 80 for t in txts])


class PaperLoader:
    def __init__(self, file_handler: FileHandler = None, col_ops_module=col_ops):
        self.col_ops = col_ops_module
        self.file_handler = file_handler or FileHandler()
        self.formatter = PaperFormatter()

    def get_paper_txts_from_collection_id(self, collection_id: int) -> List[str]:
        paper_paths = self.col_ops.get_paper_paths_from_collection_id(collection_id)
        markdown_paths = [self.file_handler.convert_pdf_path_to_md_fname(p) for p in paper_paths]
        return [self.file_handler.load_markdown_paper(p) for p in markdown_paths]

    def get_context_from_collection(self, name: str) -> str:
        collection_id = self.col_ops.get_collection_id(name)
        paper_txts = self.get_paper_txts_from_collection_id(collection_id)
        metadatas = self.col_ops.get_metadata_from_collection_id(collection_id)
        paper_contexts = [
            self.formatter.combine_metadata_and_txt(m, t) for m, t in zip(metadatas, paper_txts)
        ]
        return self.formatter.add_separator_to_txts(paper_contexts)


def get_context_from_collection(
    name: str, col_ops_module=col_ops, file_loader: Callable[[str], str] = None
) -> str:
    warnings.warn(
        "This function is deprecated. Use PaperLoader.get_context_from_collection() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    paper_loader = PaperLoader(col_ops_module)
    return paper_loader.get_context_from_collection(name)
