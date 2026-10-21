"""ChromaDB index over the KB chunks and the ``retrieve`` function (PROJECT_FLOW 3.6)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from src.llm.embed import QUERY_PREFIX, embed
from src.rag.chunking import Chunk, build_chunks

INDEX_DIR = Path("kb/index/chroma_nomic")
DEFAULT_COLLECTION = "kb_w150"
# Inputs the deployed system does not have (it has no NDVI feed at advisory time; ADR-035).
# A comma-separated clause that names one is dropped from the passage the LLM sees; the corpus
# and the index are unchanged.
UNAVAILABLE_INPUT = re.compile(r",\s*[^,;()]*\bNDVI\b[^,;()]*")


@dataclass
class Passage:
    """A retrieved passage."""

    chunk_id: str
    section: str
    title: str
    text: str
    source_citation: str
    source_type: str
    score: float  # cosine similarity to the query


def redact_unavailable_inputs(text: str) -> str:
    """Drop clauses that refer to inputs the system lacks at runtime (see ADR-035)."""
    return UNAVAILABLE_INPUT.sub("", text)


def _citation(c: Chunk) -> str:
    urls = "; ".join(c.source_urls) if c.source_urls else "no URL (synthesis)"
    return f"{c.source_description} [{urls}]"


def build_index(chunks: list[Chunk], collection: str = DEFAULT_COLLECTION) -> None:
    """(Re)build ``collection``: embed ``title. text`` with the document prefix."""
    db = chromadb.PersistentClient(path=str(INDEX_DIR))
    existing = {c.name if hasattr(c, "name") else str(c) for c in db.list_collections()}
    if collection in existing:
        db.delete_collection(collection)
    col = db.create_collection(collection, metadata={"hnsw:space": "cosine"})
    vecs = embed([f"{c.title}. {c.text}" for c in chunks])
    col.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vecs.tolist(),
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "section": c.source_section,
                "title": c.title,
                "citation": _citation(c),
                "source_type": c.source_type,
            }
            for c in chunks
        ],
    )


def retrieve(query: str, k: int = 5, collection: str = DEFAULT_COLLECTION) -> list[Passage]:
    """Return the top-``k`` passages for ``query`` (cosine similarity)."""
    col = chromadb.PersistentClient(path=str(INDEX_DIR)).get_collection(collection)
    q = embed([query], prefix=QUERY_PREFIX)
    res: Any = col.query(query_embeddings=q.tolist(), n_results=k)
    out = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
    ):
        out.append(
            Passage(
                chunk_id=cid,
                section=meta["section"],
                title=redact_unavailable_inputs(meta["title"]),
                text=redact_unavailable_inputs(doc),
                source_citation=meta["citation"],
                source_type=meta["source_type"],
                score=1.0 - float(dist),
            )
        )
    return out


def config_chunks(name: str) -> list[Chunk]:
    """Chunk configurations compared in the benchmark: (split threshold, target, overlap)."""
    cfg = {"w100": (130, 100, 20), "w150": (200, 150, 30), "w200": (260, 200, 40)}[name]
    return build_chunks(*cfg)
