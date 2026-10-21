"""Claim-level LLM-as-judge faithfulness ("grounding") of advisory sentences (ADR-035).

RAGAS-style (Es et al. 2023, R13): each advice sentence is decomposed into atomic claims, and
each claim is judged against ONLY the passages the sentence cites. The judge (llama3, 8B) is a
different model from the writer (qwen2.5:14b) to avoid self-preference. A sentence is

* ``supported``  - every claim is supported by its cited passages;
* ``unsupported`` - no claim is supported, or the sentence cites no valid passage;
* ``partial``    - anything in between.

"Grounding" and "faithfulness" in this project mean this metric and nothing else; the cosine
score in ``cosine_relevance.py`` is a relevance diagnostic only.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.llm.cosine_relevance import strip_citations
from src.llm.embed import client

JUDGE_MODEL = "llama3:latest"
CITE = re.compile(r"\[(\d+)\]")
VERDICTS = ("supported", "partially_supported", "unsupported")

DECOMPOSE_PROMPT = """Split the sentence into atomic factual claims. Each claim states ONE fact \
(a number with its unit, an action, a timing, a condition) and must make sense on its own. Keep \
numbers, units and names exactly as written. Do not add facts.

Example sentence: Apply 120 kg nitrogen per hectare in three splits and keep 5 cm of standing water.
Example output: {{"claims": ["Apply 120 kg nitrogen per hectare.", \
"The nitrogen is applied in three splits.", "Keep 5 cm of standing water."]}}

Sentence: {sentence}
Output JSON only: {{"claims": [...]}}"""

VERDICT_PROMPT = """You are a strict fact checker. Decide whether the CLAIM is supported by the \
PASSAGES, using only the passages.

Rubric:
- supported: the passages state the claim, with the same numbers, units, stages and meaning.
- partially_supported: part of the claim is stated, or the passages imply it but a number, unit, \
condition or detail in the claim is missing from them.
- unsupported: the passages do not state it, contradict it, or use a number or abbreviation with a \
different meaning (for example 60 days vs 60 kg).

PASSAGES:
{passages}

CLAIM: {claim}
Output JSON only: {{"verdict": "supported" | "partially_supported" | "unsupported", \
"reason": "<=20 words"}}"""


@dataclass
class ClaimVerdict:
    """Verdict for one atomic claim."""

    claim: str
    verdict: str
    reason: str


@dataclass
class SentenceVerdict:
    """Verdict for one advice sentence."""

    sentence: str
    cited: list[int]
    claims: list[ClaimVerdict] = field(default_factory=list)
    verdict: str = "unsupported"


@dataclass
class Faithfulness:
    """LLM-judge faithfulness of one advisory."""

    sentences: list[SentenceVerdict]
    frac_supported: float
    frac_partial: float
    frac_unsupported: float
    claim_supported: float  # share of atomic claims judged supported (RAGAS-style)
    judge_model: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return asdict(self)


def _ask(prompt: str, num_predict: int = 300) -> dict[str, Any]:
    resp = client().chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "seed": 42, "num_ctx": 4096, "num_predict": num_predict},
        format="json",
        keep_alive="30m",
    )
    try:
        out = json.loads(resp["message"]["content"])
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


def decompose(sentence: str) -> list[str]:
    """Atomic claims of ``sentence`` (the sentence itself if the judge returns nothing usable)."""
    plain = strip_citations(sentence)
    claims = _ask(DECOMPOSE_PROMPT.format(sentence=plain)).get("claims")
    if isinstance(claims, list):
        out = [str(c).strip() for c in claims if str(c).strip()]
        if out:
            return out[:6]
    return [plain]


def judge_claim(claim: str, passages: list[tuple[int, str]]) -> ClaimVerdict:
    """Verdict for ``claim`` against the cited ``passages``; unparseable output is unsupported."""
    body = "\n\n".join(f"[{n}] {p}" for n, p in passages)
    out = _ask(VERDICT_PROMPT.format(passages=body, claim=claim), 120)
    v = (
        str(out.get("verdict", ""))
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("partly", "partially_supported")
    )
    if v == "partial":
        v = "partially_supported"
    if v not in VERDICTS:
        return ClaimVerdict(claim, "unsupported", "unparseable judge output")
    return ClaimVerdict(claim, v, str(out.get("reason", ""))[:200])


def judge_sentence(sentence: str, passage_texts: list[str]) -> SentenceVerdict:
    """Judge one advice sentence against the passages it cites (indices into ``passage_texts``)."""
    cited = sorted({int(n) for n in CITE.findall(sentence) if 1 <= int(n) <= len(passage_texts)})
    sv = SentenceVerdict(sentence=sentence, cited=cited)
    if not cited:
        return sv
    texts = [(n, passage_texts[n - 1]) for n in cited]
    sv.claims = [judge_claim(c, texts) for c in decompose(sentence)]
    verdicts = [c.verdict for c in sv.claims]
    if all(v == "supported" for v in verdicts):
        sv.verdict = "supported"
    elif all(v == "unsupported" for v in verdicts):
        sv.verdict = "unsupported"
    else:
        sv.verdict = "partial"
    return sv


def judge_advisory(sentences: list[str], passage_texts: list[str]) -> Faithfulness:
    """Faithfulness of all advice ``sentences`` against the numbered ``passage_texts``."""
    svs = [judge_sentence(s, passage_texts) for s in sentences]
    n = max(len(svs), 1)
    claims = [c for sv in svs for c in sv.claims]
    return Faithfulness(
        sentences=svs,
        frac_supported=sum(sv.verdict == "supported" for sv in svs) / n,
        frac_partial=sum(sv.verdict == "partial" for sv in svs) / n,
        frac_unsupported=sum(sv.verdict == "unsupported" for sv in svs) / n,
        claim_supported=(sum(c.verdict == "supported" for c in claims) / len(claims))
        if claims
        else 0.0,
        judge_model=JUDGE_MODEL,
    )
