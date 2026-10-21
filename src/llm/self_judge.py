"""LLM-as-judge faithfulness check of generated advice sentences (calibration for ADR-034).

The judge is the same local model that wrote the advisories (qwen2.5:14b), so it is biased toward
its own output; it is used only to see how the cosine score relates to a stricter, semantic
reading, and a sample of its verdicts is checked by hand in the report.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.llm.advisor import PRIMARY_MODEL
from src.llm.embed import client

RUNS = Path("logs/advisory_runs_full.jsonl")
OUT = Path("models/results/self_judge_glossary_run.json")
CITE = re.compile(r"\[(\d+)\]")

PROMPT = """You check whether a sentence of farm advice is supported by numbered source passages.
Passages:
{passages}

Sentence: {sentence}

Is every factual claim in the sentence (numbers, units, crops, stages, actions) stated by or \
directly implied by the passages above? Answer with JSON only: \
{{"verdict": "supported" | "partly" | "unsupported", "reason": "<=20 words"}}"""


def judge(sentence: str, passages: list[str]) -> dict[str, Any]:
    """Return the judge's verdict for ``sentence`` against ``passages``."""
    body = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    resp = client().chat(
        model=PRIMARY_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(passages=body, sentence=sentence)}],
        options={"temperature": 0.0, "seed": 42, "num_ctx": 4096, "num_predict": 120},
        format="json",
        keep_alive="30m",
    )
    try:
        out: dict[str, Any] = json.loads(resp["message"]["content"])
        if out.get("verdict") not in {"supported", "partly", "unsupported"}:
            out["verdict"] = "unsupported"
        return out
    except json.JSONDecodeError:
        return {"verdict": "unsupported", "reason": "unparseable judge output"}


def main() -> None:
    """Judge every advice sentence against its cited passages and against all retrieved ones."""
    rows = []
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        run = json.loads(line)
        texts = [f"{p['title']}. {p['text']}" for p in run["passages"]]
        for s, sim, best in run["sentence_sims"]:
            cited = sorted({int(n) for n in CITE.findall(s) if 1 <= int(n) <= len(texts)})
            v_all = judge(s, texts)
            v_cited = (
                judge(s, [texts[n - 1] for n in cited]) if cited else {"verdict": "unsupported"}
            )
            rows.append(
                {
                    "run": run["i"],
                    "district": run["district"],
                    "scenario": run["scenario"],
                    "analog_year": run["analog_year"],
                    "sentence": s,
                    "cited": cited,
                    "max_cosine": sim,
                    "best_passage": best,
                    "judge_any": v_all["verdict"],
                    "judge_cited": v_cited["verdict"],
                    "judge_reason": v_all.get("reason", ""),
                }
            )
        print("run", run["i"], flush=True)
    sup_any = np.array([r["judge_any"] == "supported" for r in rows])
    sup_cited = np.array([r["judge_cited"] == "supported" for r in rows])
    cos = np.array([r["max_cosine"] for r in rows])
    sweep = []
    for t in np.round(np.arange(0.5, 0.9, 0.02), 2):
        pred = cos >= t
        sweep.append(
            {
                "threshold": float(t),
                "share_relevant": float(pred.mean()),
                "agree_with_judge_any": float((pred == sup_any).mean()),
                "relevant_but_judged_unsupported": int((pred & ~sup_any).sum()),
                "unrelevant_but_judged_supported": int((~pred & sup_any).sum()),
            }
        )
    summary = {
        "n_sentences": len(rows),
        "judge_supported_any": float(sup_any.mean()),
        "judge_supported_cited": float(sup_cited.mean()),
        "verdict_counts_any": {
            v: int(sum(r["judge_any"] == v for r in rows))
            for v in ("supported", "partly", "unsupported")
        },
        "mean_cosine_supported": float(cos[sup_any].mean()) if sup_any.any() else None,
        "mean_cosine_not_supported": float(cos[~sup_any].mean()) if (~sup_any).any() else None,
        "sweep": sweep,
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "sweep"}, indent=1))


if __name__ == "__main__":
    main()
