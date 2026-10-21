"""Post-hoc cosine_relevance check (ADR-010, PROJECT_FLOW 3.8).

For each advisory sentence take the maximum cosine similarity (nomic-embed-text, ADR-008) to the
retrieved passages. A sentence is relevant when that maximum is at least ``threshold``. The score
is the share of relevant sentences; the mean maximum cosine is reported alongside it.
Sentences and passages are both embedded with the document prefix (a symmetric comparison).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from src.llm.embed import embed
from src.rag.retriever import Passage

THRESHOLD = 0.78  # ADR-010 started at 0.6; validated on 61 labelled sentences in ADR-034
CITATION = re.compile(r"\s*\[\d+(?:\s*[,;]\s*\d+)*\]")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


@dataclass
class CosineRelevanceResult:
    """Per-sentence maximum similarities and the aggregate scores."""

    sentences: list[str]
    max_sim: list[float]
    best_passage: list[str]
    relevant_fraction: float
    mean_max_cosine: float


def strip_citations(text: str) -> str:
    """Remove inline ``[n]`` markers."""
    return CITATION.sub("", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries; citations stay attached to their sentence."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•").strip()
        if line:
            out.extend(s.strip() for s in SENTENCE.split(line) if s.strip())
    return out


def passage_vectors(passages: list[Passage]) -> np.ndarray:
    """Embed passages as ``title. text``."""
    return embed([f"{p.title}. {p.text}" for p in passages])


def cosine_relevance_detail(
    sentences: list[str],
    passages: list[Passage],
    threshold: float = THRESHOLD,
    pvecs: np.ndarray | None = None,
) -> CosineRelevanceResult:
    """Score every sentence against ``passages``."""
    if not sentences or not passages:
        return CosineRelevanceResult([], [], [], 0.0, 0.0)
    pv = pvecs if pvecs is not None else passage_vectors(passages)
    sv = embed([strip_citations(s) for s in sentences])
    sims = sv @ pv.T
    mx = sims.max(axis=1)
    return CosineRelevanceResult(
        sentences=sentences,
        max_sim=[float(v) for v in mx],
        best_passage=[passages[int(i)].chunk_id for i in sims.argmax(axis=1)],
        relevant_fraction=float((mx >= threshold).mean()),
        mean_max_cosine=float(mx.mean()),
    )


def cosine_relevance_score(
    advisory_sentences: list[str], retrieved_passages: list[Passage], threshold: float = THRESHOLD
) -> float:
    """Return the share of sentences whose best passage similarity is >= ``threshold``."""
    return cosine_relevance_detail(
        advisory_sentences, retrieved_passages, threshold
    ).relevant_fraction
