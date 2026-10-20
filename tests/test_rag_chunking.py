"""Chunking and provenance mapping for the KB corpus (no Ollama needed)."""

from src.rag.chunking import build_chunks, parse_corpus, parse_sources, split_text, words


def test_every_subsection_becomes_a_chunk() -> None:
    items = parse_corpus()
    chunks = build_chunks()
    assert len(items) == 54
    assert {c.source_section for c in chunks} == {i["section"] for i in items}
    assert all(c.n_words <= 200 for c in chunks)


def test_source_mapping_and_types() -> None:
    src = parse_sources()
    assert any("wikipedia.org/wiki/Rice_blast" in u for _, u in src["D1"])
    assert {c.chunk_id: c.source_type for c in build_chunks()}["J2"] == "synthesis"
    assert {c.chunk_id: c.source_type for c in build_chunks()}["D1"] == "wikipedia"


def test_split_text_overlaps() -> None:
    text = " ".join(f"Sentence number {i} has exactly seven words here." for i in range(40))
    parts = split_text(text, target=60, overlap=15)
    assert len(parts) > 1
    assert all(words(p) <= 75 for p in parts)
    assert any(s in parts[1] for s in parts[0].split(". ")[-2:])
