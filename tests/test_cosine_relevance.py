"""CosineRelevance score arithmetic with a fake embedder (no Ollama needed)."""

import numpy as np
import pytest
from src.llm import cosine_relevance
from src.llm.advisor import parse
from src.rag.retriever import Passage


def _fake_embed(texts: list[str], prefix: str = "", batch: int = 32) -> np.ndarray:
    """One-hot-ish vectors: 'alpha' texts point one way, 'beta' texts another."""
    out = [[1.0, 0.0] if "alpha" in t else [0.0, 1.0] for t in texts]
    return np.asarray(out, dtype=np.float32)


def _passage(text: str) -> Passage:
    return Passage("X1", "X1", "t", text, "src", "synthesis", 0.0)


def test_relevant_fraction_and_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cosine_relevance, "embed", _fake_embed)
    ps = [_passage("alpha passage")]
    res = cosine_relevance.cosine_relevance_detail(
        ["alpha claim [1].", "beta claim [1]."], ps, threshold=0.6
    )
    assert res.max_sim == pytest.approx([1.0, 0.0])
    assert res.relevant_fraction == 0.5
    assert cosine_relevance.cosine_relevance_score(["alpha claim"], ps) == 1.0


def test_empty_inputs() -> None:
    assert cosine_relevance.cosine_relevance_detail([], []).relevant_fraction == 0.0


def test_strip_and_split() -> None:
    assert cosine_relevance.strip_citations("Apply zinc [2][3].") == "Apply zinc."
    assert cosine_relevance.split_sentences(
        "- One thing [1]. Another thing [2].\n- Third [3]."
    ) == [
        "One thing [1].",
        "Another thing [2].",
        "Third [3].",
    ]


def test_parse_drops_format_echo() -> None:
    text = "Scenario: dry year.\nAdvice:\n(4 to 6 bullets)\n- Irrigate [1]."
    scenario, sents = parse(text)
    assert scenario == "dry year."
    assert sents == ["Irrigate [1]."]
