"""Judge the 30 scenario advisories with the llama3 claim-level judge and compare to cosine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.llm.cosine_relevance import THRESHOLD
from src.llm.judge import judge_advisory

RUNS = Path("logs/advisory_runs_full.jsonl")
OUT = Path("models/results/advisory_faithfulness.json")


def main() -> None:
    """Judge every advisory, then write per-run and aggregate faithfulness next to cosine."""
    rows: list[dict[str, Any]] = []
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        texts = [f"{p['title']}. {p['text']}" for p in r["passages"]]
        sents = [s for s, _, _ in r["sentence_sims"]]
        f = judge_advisory(sents, texts)
        sims = [x[1] for x in r["sentence_sims"]]
        rows.append(
            {
                "i": r["i"],
                "district": r["district"],
                "scenario": r["scenario"],
                "analog_year": r["analog_year"],
                "frac_supported": f.frac_supported,
                "frac_partial": f.frac_partial,
                "frac_unsupported": f.frac_unsupported,
                "claim_supported": f.claim_supported,
                "cosine_relevance": r["cosine_relevance"],
                "n_sentences": len(sents),
                "max_cosine": sims,
                "faithfulness": f.as_dict(),
            }
        )
        print(
            f"run {r['i']:2d} sup {f.frac_supported:.2f} part {f.frac_partial:.2f} "
            f"unsup {f.frac_unsupported:.2f} (cos {r['cosine_relevance']:.2f})",
            flush=True,
        )
    sup = np.array([x["frac_supported"] for x in rows])
    par = np.array([x["frac_partial"] for x in rows])
    uns = np.array([x["frac_unsupported"] for x in rows])
    cos = np.array([x["cosine_relevance"] for x in rows])
    flat = [
        (sv["verdict"], c)
        for x in rows
        for sv, c in zip(x["faithfulness"]["sentences"], x["max_cosine"], strict=True)
    ]
    cs = np.array([c for v, c in flat if v == "supported"])
    cn = np.array([c for v, c in flat if v != "supported"])
    summary = {
        "n_runs": len(rows),
        "n_sentences": len(flat),
        "mean_frac_supported": float(sup.mean()),
        "mean_frac_partial": float(par.mean()),
        "mean_frac_unsupported": float(uns.mean()),
        "mean_claim_supported": float(np.mean([x["claim_supported"] for x in rows])),
        "runs_below_70pct_supported": [x["i"] for x in rows if x["frac_supported"] < 0.7],
        "by_scenario": {
            s: float(np.mean([x["frac_supported"] for x in rows if x["scenario"] == s]))
            for s in ("bad", "average", "good")
        },
        "cosine_relevance_mean": float(cos.mean()),
        "cosine_threshold": THRESHOLD,
        "mean_cosine_judged_supported": float(cs.mean()) if len(cs) else None,
        "mean_cosine_judged_not_supported": float(cn.mean()) if len(cn) else None,
        "sentence_level_agreement": float(
            np.mean([(c >= THRESHOLD) == (v == "supported") for v, c in flat])
        ),
    }
    OUT.write_text(json.dumps({"summary": summary, "runs": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
