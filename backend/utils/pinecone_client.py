from functools import lru_cache
from typing import Iterable, List

from pinecone import Pinecone, ServerlessSpec

from backend.config import get_settings


@lru_cache(maxsize=1)
def _pc() -> Pinecone:
    settings = get_settings()
    return Pinecone(api_key=settings["pinecone_api_key"])


def _list_index_names(pc: Pinecone) -> List[str]:
    """Return all index names regardless of Pinecone client version."""
    index_listing = pc.list_indexes()
    if hasattr(index_listing, "names"):
        return list(index_listing.names())
    if hasattr(index_listing, "indexes"):
        return [idx["name"] for idx in index_listing.indexes]
    names: List[str] = []
    for item in index_listing:
        if isinstance(item, str):
            names.append(item)
        elif hasattr(item, "name"):
            names.append(item.name)
        elif isinstance(item, dict) and "name" in item:
            names.append(item["name"])
    return names


@lru_cache(maxsize=1)
def get_index():
    settings = get_settings()
    pc = _pc()
    index_name = settings["pinecone_index"]

    existing_indexes = _list_index_names(pc)
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=settings["pinecone_dimension"],
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings["pinecone_cloud"],
                region=settings["pinecone_region"]
            ),
        )
    return pc.Index(index_name)


def insert_vectors(
    index,
    embeddings: List[List[float]],
    metadata_items: Iterable[dict],
    id_prefix: str = "chunk"
) -> None:
    vectors = []
    for i, metadata in enumerate(metadata_items):
        vectors.append({
            "id": f"{id_prefix}_{i}",
            "values": embeddings[i],
            "metadata": metadata
        })
    index.upsert(vectors)
