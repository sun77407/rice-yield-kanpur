"""Chunk ``kb/raw/kb_corpus.txt`` along its ``###`` subsections (ADR-033)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

CORPUS = Path("kb/raw/kb_corpus.txt")
SOURCES = Path("kb/raw/SOURCES.md")
OUT = Path("kb/chunks/kb.jsonl")
HEADING = re.compile(r"^###\s+([A-J]\d+)\.\s+(.*)$")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


@dataclass
class Chunk:
    """One retrievable passage with provenance."""

    chunk_id: str
    source_section: str
    title: str
    text: str
    source_description: str
    source_urls: list[str]
    source_type: str  # other_source | wikipedia | synthesis
    n_words: int


def words(text: str) -> int:
    """Whitespace word count."""
    return len(text.split())


def parse_sources(path: Path = SOURCES) -> dict[str, list[tuple[str, str]]]:
    """Map section id (e.g. ``B7``) to the (source name, url) rows that cite it."""
    out: dict[str, list[tuple[str, str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or not cells[1].startswith("http"):
            continue
        name, url, secs = cells
        for tok in re.split(r"\s*,\s*", secs):
            m = re.fullmatch(r"([A-J])(\d+)[–-]([A-J])?(\d+)", tok)
            if m:
                for i in range(int(m.group(2)), int(m.group(4)) + 1):
                    out.setdefault(f"{m.group(1)}{i}", []).append((name, url))
            elif re.fullmatch(r"[A-J]\d+", tok):
                out.setdefault(tok, []).append((name, url))
    return out


def parse_corpus(path: Path = CORPUS) -> list[dict[str, str]]:
    """Return ``{section, title, source_line, body}`` for every ``###`` subsection."""
    items: list[dict[str, str]] = []
    cur: dict[str, list[str] | str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## END OF CORPUS"):
            break
        m = HEADING.match(line)
        if m:
            cur = {"section": m.group(1), "title": m.group(2).strip(), "lines": []}
            items.append(cur)  # type: ignore[arg-type]
        elif line.startswith("#") or line.strip() == "---":
            cur = None
        elif cur is not None and line.strip():
            cur["lines"].append(line.strip())  # type: ignore[union-attr]
    out = []
    for it in items:
        lines = it["lines"]
        src = lines[0] if lines and re.match(r"Sources?:", lines[0]) else ""
        body = " ".join(lines[1:] if src else lines)
        out.append(
            {
                "section": str(it["section"]),
                "title": str(it["title"]),
                "source_line": src,
                "body": body,
            }
        )
    return out


def split_text(body: str, target: int, overlap: int) -> list[str]:
    """Greedy sentence packing to ~``target`` words, carrying ~``overlap`` words forward."""
    sents = SENTENCE.split(body)
    chunks: list[list[str]] = []
    cur: list[str] = []
    for s in sents:
        if cur and words(" ".join(cur)) + words(s) > target:
            chunks.append(cur)
            carry: list[str] = []
            for prev in reversed(cur):
                if words(" ".join([prev, *carry])) > overlap:
                    break
                carry.insert(0, prev)
            cur = carry
        cur.append(s)
    if cur and (not chunks or cur != chunks[-1][-len(cur) :]):
        chunks.append(cur)
    return [" ".join(c) for c in chunks]


def source_type(rows: list[tuple[str, str]]) -> str:
    """Return ``synthesis``, ``wikipedia`` (only Wikipedia) or ``other_source``."""
    if not rows:
        return "synthesis"
    if all("wikipedia" in n.lower() for n, _ in rows):
        return "wikipedia"
    return "other_source"


def build_chunks(max_words: int = 200, target: int = 150, overlap: int = 30) -> list[Chunk]:
    """Chunk the corpus; subsections above ``max_words`` are split."""
    sources = parse_sources()
    chunks: list[Chunk] = []
    for it in parse_corpus():
        rows = sources.get(it["section"], [])
        parts = (
            split_text(it["body"], target, overlap)
            if words(it["body"]) > max_words
            else [it["body"]]
        )
        for i, text in enumerate(parts):
            cid = it["section"] if len(parts) == 1 else f"{it['section']}-{i + 1}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    source_section=it["section"],
                    title=it["title"],
                    text=text,
                    source_description=it["source_line"].split(":", 1)[-1].strip().rstrip(".")
                    or "synthesis by corpus author",
                    source_urls=sorted({u for _, u in rows}),
                    source_type=source_type(rows),
                    n_words=words(text),
                )
            )
    return chunks


def write_chunks(chunks: list[Chunk], path: Path = OUT) -> None:
    """Write JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(c)) for c in chunks) + "\n", encoding="utf-8")


def main() -> None:
    """Chunk with the default configuration and print a summary."""
    ch = build_chunks()
    write_chunks(ch)
    types = {
        t: sum(c.source_type == t for c in ch) for t in ("other_source", "wikipedia", "synthesis")
    }
    w = [c.n_words for c in ch]
    print(
        f"{len(ch)} chunks; words min/median/max = "
        f"{min(w)}/{sorted(w)[len(w) // 2]}/{max(w)}; {types}"
    )
    print("split sections:", sorted({c.source_section for c in ch if "-" in c.chunk_id}))


if __name__ == "__main__":
    main()
