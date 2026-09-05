"""Vector store: embed chunks locally and retrieve the closest ones.

Chroma's default embedding function is all-MiniLM-L6-v2 running on
onnxruntime. That matters for two reasons: embedding costs nothing and needs
no API key, and it avoids pulling in PyTorch, which is too large for most
free hosting tiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from .ingest import Chunk

DB_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION = "notes"


@dataclass(frozen=True)
class Retrieved:
    """A chunk that came back from a search, with its distance score."""

    text: str
    source: str
    page: int
    distance: float


def get_collection():
    """Open (or create) the on-disk collection.

    Cosine distance is the right metric for sentence embeddings; Chroma
    defaults to squared L2, so it is set explicitly here.
    """
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[Chunk], batch_size: int = 100) -> int:
    """Embed and store chunks. Returns how many were written.

    Chunk IDs are content hashes, so upserting the same document twice
    overwrites rather than duplicating — re-indexing is safe to repeat.
    """
    if not chunks:
        return 0

    collection = get_collection()

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[{"source": c.source, "page": c.page} for c in batch],
        )

    return len(chunks)


def search(question: str, k: int = 5) -> list[Retrieved]:
    """Return the k chunks closest to the question."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    result = collection.query(
        query_texts=[question],
        n_results=min(k, collection.count()),
    )

    hits: list[Retrieved] = []
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    for text, meta, distance in zip(documents, metadatas, distances):
        hits.append(
            Retrieved(
                text=text,
                source=str(meta.get("source", "unknown")),
                page=int(meta.get("page", 0)),
                distance=float(distance),
            )
        )
    return hits


def stats() -> dict:
    """Chunk count and the list of indexed documents, for the sidebar."""
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return {"chunks": 0, "documents": []}

    everything = collection.get(include=["metadatas"])
    sources = sorted({str(m.get("source", "unknown")) for m in everything["metadatas"]})
    return {"chunks": count, "documents": sources}


def reset() -> None:
    """Drop the whole index."""
    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # already gone
