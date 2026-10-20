"""Advisory generation with a local LLM over retrieved KB passages (PROJECT_FLOW 3.7).

Only Ollama is used (ADR-018: no cloud LLMs). Primary model ``qwen2.5:14b-instruct-q6_K``,
fallback ``llama3:latest`` (ADR-007). The prompt allows the model to restate the scenario
facts it is given in one separate "Scenario:" paragraph; every sentence under "Advice:" must
cite a numbered passage, and only those sentences are grounding-checked.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src.llm.embed import client
from src.llm.grounding import (
    THRESHOLD,
    GroundingResult,
    grounding_detail,
    passage_vectors,
    split_sentences,
    strip_citations,
)
from src.rag.retriever import Passage, retrieve

PRIMARY_MODEL = os.environ.get("LLM_PRIMARY") or "qwen2.5:14b-instruct-q6_K"
FALLBACK_MODEL = os.environ.get("LLM_FALLBACK") or "llama3:latest"
LOG = Path("logs/advisory_generation.jsonl")
INSUFFICIENT = "Insufficient information in local knowledge base."
CITE = re.compile(r"\[(\d+)\]")
TIMEOUT_S = 120
# The corpus uses abbreviations it never defines (J3: "nitrogen split dose at 60 DAT"); without
# this the model read "60 DAT" as "60 kg" in 7 of 30 first-run advisories (see ADR-034).
GLOSSARY = (
    "GLOSSARY: DAT = days after transplanting; BPH = brown planthopper; "
    "NPK = nitrogen, phosphorus and potash fertilizer; Zn = zinc; "
    "PMFBY = crop insurance scheme; FYM = farmyard manure; MSP = minimum support price; "
    "AWD = alternate wetting and drying."
)

SYSTEM_PROMPT = (
    "You are an agronomy advisory assistant for rice farmers in Kanpur, Uttar Pradesh.\n"
    "Use ONLY the following numbered passages. For every factual claim, cite the passage "
    "number [n]. "
    f"If the passages do not cover a question, say '{INSUFFICIENT}' "
    "Do not invent facts or draw on general knowledge.\n\n"
    "Write exactly this format and nothing else:\n"
    "Scenario: <one or two sentences restating only the SCENARIO FACTS given by the user; "
    "no advice>\n"
    "Advice:\n"
    "- <one sentence of advice that ends with its citation, for example [2] or [1][3]>\n"
    "- <next sentence of advice with its citation>\n\n"
    "Write 4 to 6 advice bullets covering fertilizer, irrigation, pest and disease watch, "
    "and what to change for this scenario.\n"
    "Rules: each bullet is one sentence and quotes numbers, dates and names only as they "
    "appear in the passage it cites. Do not mention yield figures in the Advice bullets. "
    "Do not use a passage number that was not given."
)


@dataclass
class YieldPrediction:
    """Inputs for one advisory: a scenario (historical analog year) and the model estimate."""

    district: str
    scenario: str  # bad | average | good
    analog_year: int
    planting_date: str
    predicted_yield_t_ha: float
    trend_reference_t_ha: float
    weather: dict[str, float | str] = field(default_factory=dict)


@dataclass
class Advisory:
    """A generated advisory with its checks."""

    text: str
    scenario_text: str
    sentences: list[str]
    citations: dict[int, list[int]]  # advice-sentence index -> passage numbers
    passages: list[Passage]
    grounding: GroundingResult
    grounding_score: float  # share of advice sentences with max cosine >= THRESHOLD
    grounding_at_060: float
    n_distinct_citations: int
    uncited_sentences: list[int]
    invalid_citations: list[int]
    insufficient: bool
    model: str
    used_fallback: bool
    latency_s: float
    retrieval_s: float
    generation_s: float


def scenario_queries(p: YieldPrediction) -> list[str]:
    """One main query (PROJECT_FLOW 2.8) plus topic and scenario sub-queries."""
    main = (
        f"Rice advisory for {p.district}, planting {p.planting_date}, "
        f"{p.scenario} year scenario. What should the farmer do about fertilizer, "
        "irrigation, pest watch?"
    )
    scen = {
        "bad": "advice for a low expected yield bad year with below-normal monsoon and heat stress",
        "good": "advice for a good year with above-normal monsoon and vigorous crop",
        "average": "advice for an average year with monsoon on time",
    }[p.scenario]
    return [
        main,
        scen,
        "fertilizer nitrogen phosphorus potash dose and timing for kharif rice",
        "irrigation standing water depth critical flowering stage",
        "pest and disease watch brown planthopper blast sheath blight humid weather",
    ]


def retrieve_for_scenario(p: YieldPrediction, per_query: int = 2, cap: int = 6) -> list[Passage]:
    """Union of the top results of each sub-query, best score first, at most ``cap`` passages."""
    best: dict[str, Passage] = {}
    for q in scenario_queries(p):
        for ps in retrieve(q, k=per_query):
            if ps.chunk_id not in best or ps.score > best[ps.chunk_id].score:
                best[ps.chunk_id] = ps
    return sorted(best.values(), key=lambda x: -x.score)[:cap]


def build_user_prompt(p: YieldPrediction, passages: list[Passage]) -> str:
    """Scenario facts (not citable) followed by the numbered passages."""
    w = p.weather
    facts = [
        f"District: {p.district}; planting date: {p.planting_date}; scenario: {p.scenario} year "
        f"(analog of Kharif {p.analog_year}).",
        f"Kharif rainfall {w.get('kharif_rainfall_mm')} mm ({w.get('rain_vs_mean')} vs district "
        f"mean); monsoon onset {w.get('monsoon_onset')}; {w.get('dry_spells')} dry spell(s) of 7+ "
        f"days; {w.get('heat_stress_days')} days above 35 C between Aug 15 and Sep 15.",
        f"Model-estimated district yield {p.predicted_yield_t_ha:.2f} t/ha "
        f"(time-trend reference {p.trend_reference_t_ha:.2f} t/ha; the model is uncertain).",
    ]
    blocks = [
        f"[{i}] {ps.title}. {ps.text}\n(Source: {ps.source_citation})"
        for i, ps in enumerate(passages, 1)
    ]
    return (
        GLOSSARY
        + "\n\nSCENARIO FACTS:\n"
        + "\n".join(facts)
        + "\n\nPASSAGES:\n"
        + "\n\n".join(blocks)
    )


def _chat(model: str, user: str) -> str:
    resp = client().chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        options={"temperature": 0.2, "seed": 42, "num_ctx": 4096, "num_predict": 600},
        keep_alive="30m",
    )
    return str(resp["message"]["content"])


def generate_text(user: str) -> tuple[str, str, bool]:
    """Call the primary model, falling back to llama3 on error or timeout (> 120 s)."""
    t0 = time.time()
    try:
        out = _chat(PRIMARY_MODEL, user)
        if time.time() - t0 > TIMEOUT_S:
            raise TimeoutError("primary model exceeded 120 s")
        return out, PRIMARY_MODEL, False
    except Exception as exc:
        _log({"event": "llm_fallback", "error": repr(exc)[:200]})
        return _chat(FALLBACK_MODEL, user), FALLBACK_MODEL, True


def parse(text: str) -> tuple[str, list[str]]:
    """Split the model output into the scenario paragraph and the advice sentences."""
    m = re.search(r"Scenario:\s*(.*?)\s*Advice:\s*(.*)", text, flags=re.S | re.I)
    scenario, advice = (m.group(1), m.group(2)) if m else ("", text)
    sents = [
        x for x in split_sentences(advice) if not re.match(r"^\(?\d+ to \d+ (advice )?bullets", x)
    ]
    return scenario.strip(), sents


def generate_advisory(
    prediction: YieldPrediction,
    retrieved: list[Passage] | None = None,
    model: str | None = None,
    threshold: float = THRESHOLD,
) -> Advisory:
    """Retrieve (unless given), generate, and run the citation and grounding checks."""
    t0 = time.time()
    passages = retrieved if retrieved is not None else retrieve_for_scenario(prediction)
    t_ret = time.time() - t0
    if not passages:
        text = f"Scenario: -\nAdvice:\n- {INSUFFICIENT}"
        used, fb, t_gen = "none", False, 0.0
    else:
        user = build_user_prompt(prediction, passages)
        t1 = time.time()
        if model:
            text, used, fb = _chat(model, user), model, False
        else:
            text, used, fb = generate_text(user)
        t_gen = time.time() - t1
    scenario, sents = parse(text)
    insufficient = [i for i, s in enumerate(sents) if INSUFFICIENT.lower() in s.lower()]
    check = [s for i, s in enumerate(sents) if i not in insufficient]
    cites = {i: [int(n) for n in CITE.findall(s)] for i, s in enumerate(check)}
    n_pass = len(passages)
    invalid = sorted({n for v in cites.values() for n in v if not 1 <= n <= n_pass})
    pvecs = passage_vectors(passages) if passages else None
    g = grounding_detail(check, passages, threshold, pvecs)
    g060 = grounding_detail(check, passages, 0.6, pvecs)
    adv = Advisory(
        text=text,
        scenario_text=scenario,
        sentences=check,
        citations=cites,
        passages=passages,
        grounding=g,
        grounding_score=g.grounded_fraction,
        grounding_at_060=g060.grounded_fraction,
        n_distinct_citations=len({n for v in cites.values() for n in v if 1 <= n <= n_pass}),
        uncited_sentences=[i for i, v in cites.items() if not v],
        invalid_citations=invalid,
        insufficient=bool(insufficient),
        model=used,
        used_fallback=fb,
        latency_s=time.time() - t0,
        retrieval_s=t_ret,
        generation_s=t_gen,
    )
    _log(
        {
            "event": "advisory",
            "district": prediction.district,
            "scenario": prediction.scenario,
            "analog_year": prediction.analog_year,
            "model": used,
            "grounding": adv.grounding_score,
            "grounding_at_0.6": adv.grounding_at_060,
            "mean_max_cosine": g.mean_max_cosine,
            "distinct_citations": adv.n_distinct_citations,
            "latency_s": round(adv.latency_s, 2),
        }
    )
    return adv


def _log(rec: dict[str, object]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"timestamp": datetime.now(UTC).isoformat(timespec="seconds"), **rec}
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


__all__ = [
    "Advisory",
    "YieldPrediction",
    "generate_advisory",
    "retrieve_for_scenario",
    "strip_citations",
]
