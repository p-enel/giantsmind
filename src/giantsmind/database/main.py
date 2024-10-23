from giantsmind.core.get_metadata import get_all_metadata_from_json
from pprint import pprint
from giantsmind.database import paper_operations as paper_ops
from giantsmind.database import collection_operations as collection_ops
from giantsmind.database.schema import init_db


if __name__ == "__main__":
    metadatas = get_all_metadata_from_json()
    for metadata in metadatas:
        metadata["paper_id"] = metadata["id"]
        metadata["authors"] = metadata["author"].split("; ")
        del metadata["id"]
        del metadata["author"]

    paper_ops.add_papers(metadatas)

    collection_ops.get_all_papers_collectionid()

    collection_ops.get_all_collections()
    collection_ops.delete_collection("results")
    paper_ids = collection_ops.get_paper_ids_from_collectionid(4)

    # paper_ops.remove_papers(paper_ids)
    # collection_ops.delete_collection("all papers")
