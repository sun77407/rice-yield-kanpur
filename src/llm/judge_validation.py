"""Check the LLM judge against the 61 hand-labelled sentences (ADR-035).

Each sample's sentences are judged against all of that sample's passages (cited as [1]..[n]);
the hand label is True (supported) or False (plausible but unsupported).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.llm.cosine_relevance_validation import SAMPLES, load_passages
from src.llm.judge import judge_sentence

OUT = Path("kb/index/judge_validation.json")


def main() -> None:
    """Judge every labelled sentence and write the confusion counts."""
    passages = load_passages()
    rows = []
    for ids, sents in SAMPLES:
        texts = [f"{passages[i].title}. {passages[i].text}" for i in ids]
        cites = "".join(f"[{n}]" for n in range(1, len(texts) + 1))
        for s, label in sents:
            v = judge_sentence(f"{s} {cites}", texts)
            rows.append({"sentence": s, "label_supported": label, "judge": v.verdict})
    probes = []
    for s in (
        "Reduce nitrogen to 60 kg per hectare in a forecast bad year [1].",
        "Make the nitrogen split dose at 60 DAT optional in a forecast bad year [1].",
    ):
        v = judge_sentence(s, [f"{passages['J3'].title}. {passages['J3'].text}"])
        probes.append(
            {
                "sentence": s,
                "judge": v.verdict,
                "expected": "unsupported" if "kg" in s else "supported",
            }
        )
    sup = [r for r in rows if r["label_supported"]]
    uns = [r for r in rows if not r["label_supported"]]
    summary = {
        "n": len(rows),
        "supported_labelled": len(sup),
        "unsupported_labelled": len(uns),
        "supported_judged_supported": sum(r["judge"] == "supported" for r in sup),
        "supported_judged_partial": sum(r["judge"] == "partial" for r in sup),
        "supported_judged_unsupported": sum(r["judge"] == "unsupported" for r in sup),
        "unsupported_judged_supported": sum(r["judge"] == "supported" for r in uns),
        "unsupported_judged_partial": sum(r["judge"] == "partial" for r in uns),
        "unsupported_judged_unsupported": sum(r["judge"] == "unsupported" for r in uns),
    }
    OUT.write_text(
        json.dumps({"summary": summary, "unit_probes": probes, "rows": rows}, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=1))
    print(json.dumps(probes, indent=1))


if __name__ == "__main__":
    main()
