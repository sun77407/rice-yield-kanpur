"""Write ``kb/index/advisory_flagged.md``: advisories below 70% grounding, for manual inspection."""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path("logs/advisory_runs_full.jsonl")
JUDGE = Path("models/results/advisory_judge.json")
OUT = Path("kb/index/advisory_flagged.md")
THRESHOLD = 0.78


def main() -> None:
    """List every advisory under 70% grounding with its failing sentences and passages."""
    verdict = {(r["run"], r["sentence"]): r for r in json.loads(JUDGE.read_text())["rows"]}
    lines = [
        "# Advisories below 70% grounding (cosine threshold 0.78)",
        "",
        "Each entry: failing sentences (max cosine to any retrieved passage, closest passage), the",
        "self-judge verdict (LLM-as-judge, same model, lenient), and the retrieved passage ids.",
        "",
    ]
    n = 0
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["grounding"] >= 0.7:
            continue
        n += 1
        ids = ", ".join(
            f"[{i + 1}] {p['chunk_id']} {p['title']}" for i, p in enumerate(r["passages"])
        )
        lines += [
            f"## Run {r['i']}: {r['district']} analog {r['analog_year']} ({r['scenario']}), "
            f"grounding {r['grounding']:.2f}, distinct citations {r['distinct_citations']}",
            f"Passages: {ids}",
            "",
        ]
        for s, sim, best in r["sentence_sims"]:
            if sim < THRESHOLD:
                v = verdict.get((r["i"], s), {})
                lines.append(
                    f"- cos {sim:.2f} (closest {best}; judge: {v.get('judge_any', '?')}): {s}"
                )
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"{n} flagged runs -> {OUT}")


if __name__ == "__main__":
    main()
