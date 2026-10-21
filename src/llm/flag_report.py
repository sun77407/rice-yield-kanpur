"""Write ``kb/index/advisory_flagged.md``: advisories under 70% fully supported."""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path("logs/advisory_runs_full.jsonl")
JUDGE = Path("models/results/advisory_faithfulness.json")
OUT = Path("kb/index/advisory_flagged.md")


def main() -> None:
    """List every advisory below 70% fully supported with its non-supported sentences/claims."""
    judged = {r["i"]: r for r in json.loads(JUDGE.read_text(encoding="utf-8"))["runs"]}
    lines = [
        "# Advisories below 70% fully-supported sentences (llama3 claim-level judge)",
        "",
        "Each sentence is split into claims and judged against only the passages it cites.",
        "Cosine relevance (threshold 0.78) is shown as a diagnostic, not as faithfulness.",
        "",
    ]
    n = 0
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        j = judged[r["i"]]
        if j["frac_supported"] >= 0.7:
            continue
        n += 1
        ids = ", ".join(
            f"[{i + 1}] {p['chunk_id']} {p['title']}" for i, p in enumerate(r["passages"])
        )
        lines += [
            f"## Run {r['i']}: {r['district']} analog {r['analog_year']} ({r['scenario']}): "
            f"supported {j['frac_supported']:.0%}, partial {j['frac_partial']:.0%}, "
            f"unsupported {j['frac_unsupported']:.0%}; "
            f"cosine relevance {j['cosine_relevance']:.0%}",
            f"Passages: {ids}",
            "",
        ]
        for sv, cos in zip(j["faithfulness"]["sentences"], j["max_cosine"], strict=True):
            if sv["verdict"] == "supported":
                continue
            lines.append(
                f"- **{sv['verdict']}** (cites {sv['cited']}, max cosine {cos:.2f}): "
                f"{sv['sentence']}"
            )
            for c in sv["claims"]:
                if c["verdict"] != "supported":
                    lines.append(f"  - {c['verdict']}: {c['claim']} ({c['reason']})")
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"{n} flagged runs -> {OUT}")


if __name__ == "__main__":
    main()
