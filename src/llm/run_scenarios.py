"""Run the 30 scenario advisories and report Phase 4 validation statistics."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.llm.advisor import generate_advisory
from src.llm.scenarios import build_scenarios

OUT = Path("models/results/advisory_runs.json")
FULL = Path("logs/advisory_runs_full.jsonl")


def main() -> None:
    """Generate one advisory per scenario, store everything, print the summary."""
    scen = build_scenarios()
    print({s: sum(x.scenario == s for x in scen) for s in ("bad", "average", "good")})
    FULL.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for i, s in enumerate(scen):
        adv = generate_advisory(s)
        row = {
            "i": i,
            "district": s.district,
            "scenario": s.scenario,
            "analog_year": s.analog_year,
            "predicted_yield": s.predicted_yield_t_ha,
            "model": adv.model,
            "used_fallback": adv.used_fallback,
            "grounding": adv.grounding_score,
            "grounding_at_0.6": adv.grounding_at_060,
            "mean_max_cosine": adv.grounding.mean_max_cosine,
            "n_sentences": len(adv.sentences),
            "distinct_citations": adv.n_distinct_citations,
            "uncited": len(adv.uncited_sentences),
            "invalid_citations": adv.invalid_citations,
            "insufficient": adv.insufficient,
            "latency_s": adv.latency_s,
            "retrieval_s": adv.retrieval_s,
            "generation_s": adv.generation_s,
        }
        rows.append(row)
        with FULL.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        **row,
                        "text": adv.text,
                        "passages": [
                            {k: v for k, v in asdict(p).items() if k != "text"} | {"text": p.text}
                            for p in adv.passages
                        ],
                        "sentence_sims": list(
                            zip(
                                adv.sentences,
                                adv.grounding.max_sim,
                                adv.grounding.best_passage,
                                strict=True,
                            )
                        ),
                    }
                )
                + "\n"
            )
        print(
            f"{i:2d} {s.district[7:]:6s} {s.analog_year} {s.scenario:7s} "
            f"g {adv.grounding_score:.2f} "
            f"(@0.6 {adv.grounding_at_060:.2f}) cites {adv.n_distinct_citations} "
            f"{adv.latency_s:5.1f}s {adv.model}",
            flush=True,
        )
    g = np.array([r["grounding"] for r in rows])
    g6 = np.array([r["grounding_at_0.6"] for r in rows])
    lat = np.array([r["latency_s"] for r in rows])
    summary = {
        "n_runs": len(rows),
        "grounding": {
            "mean": float(g.mean()),
            "median": float(np.median(g)),
            "p10": float(np.percentile(g, 10)),
            "min": float(g.min()),
        },
        "grounding_at_0.6": {
            "mean": float(g6.mean()),
            "median": float(np.median(g6)),
            "p10": float(np.percentile(g6, 10)),
        },
        "mean_max_cosine_mean": float(np.mean([r["mean_max_cosine"] for r in rows])),
        "latency_s": {
            "median": float(np.median(lat)),
            "p95": float(np.percentile(lat, 95)),
            "max": float(lat.max()),
            "first_call": float(lat[0]),
        },
        "with_ge_3_distinct_citations": int(sum(r["distinct_citations"] >= 3 for r in rows)),
        "runs_below_70pct_grounding": [r["i"] for r in rows if r["grounding"] < 0.7],
        "fallback_used": int(sum(r["used_fallback"] for r in rows)),
        "invalid_citation_runs": int(sum(bool(r["invalid_citations"]) for r in rows)),
        "insufficient_runs": int(sum(r["insufficient"] for r in rows)),
    }
    OUT.write_text(json.dumps({"summary": summary, "runs": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
