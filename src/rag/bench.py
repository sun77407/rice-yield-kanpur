"""Retrieval benchmark: 20 handcrafted queries with hand-labelled relevant sections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.rag.retriever import build_index, config_chunks, retrieve

OUT = Path("kb/index/retrieval_bench.json")

# (topic, query, relevant sections). Relevant = passages that contain the information needed to
# answer the query; labelled by reading kb_corpus.txt before any retrieval was run.
QUERIES: list[tuple[str, str, set[str]]] = [
    (
        "nutrient",
        "How much nitrogen, phosphorus and potash should I apply per hectare for kharif paddy?",
        {"A1", "H4", "J4"},
    ),
    (
        "nutrient",
        "When should urea be split and top-dressed after transplanting?",
        {"A2", "J4", "J3"},
    ),
    (
        "nutrient",
        "My soil pH is high and leaves show yellow patches with brown dead spots. "
        "Which micronutrient is lacking?",
        {"A4", "A3", "J1"},
    ),
    (
        "nutrient",
        "Can farmyard manure and biofertilizers replace part of the chemical fertilizer?",
        {"A6", "I2", "H4"},
    ),
    (
        "water",
        "How deep should standing water be after transplanting and when do I stop irrigating "
        "before harvest?",
        {"B4", "B6"},
    ),
    ("water", "Which growth stage is most sensitive to water shortage?", {"B5"}),
    (
        "water",
        "How can I save irrigation water by letting the field dry between irrigations?",
        {"B7"},
    ),
    (
        "water",
        "The monsoon is late this year and I need pump irrigation. What should I change?",
        {"J2", "H5", "F3", "B8"},
    ),
    (
        "pest",
        "Plants are drying out in circular patches and tillers turn yellow at the base. "
        "Which insect is responsible?",
        {"C2", "C1"},
    ),
    ("pest", "How do I control planthoppers without causing an outbreak?", {"C3", "I1"}),
    ("pest", "Which insects other than planthoppers attack paddy?", {"C4"}),
    (
        "disease",
        "Grey spindle-shaped spots on leaves and the stem breaks below the panicle. "
        "What disease is this?",
        {"D1"},
    ),
    (
        "disease",
        "Water-soaked lesions on lower leaves in humid weather. How do I manage the disease?",
        {"D4", "D5"},
    ),
    ("disease", "Which weather conditions favour blast infection?", {"D2"}),
    (
        "variety",
        "Which short-duration variety should I choose if transplanting is late?",
        {"E7", "E4", "J2"},
    ),
    ("variety", "Which paddy varieties tolerate sodic or salty soil?", {"E6", "E1"}),
    ("variety", "What are the yield and disease resistance of Narendra-80?", {"E4"}),
    (
        "establishment",
        "How old should seedlings be at transplanting and when should the nursery be sown?",
        {"F3"},
    ),
    (
        "establishment",
        "How much seed per hectare do I need and how should seed be treated?",
        {"F1", "F2"},
    ),
    ("establishment", "Does very hot weather during flowering reduce grain set?", {"F5", "I3"}),
]


def evaluate(collection: str) -> dict[str, Any]:
    """Return precision@3 and recall@5 at section level, plus per-query detail."""
    p3, r5, hit3, p1, rr, ceil, rows = [], [], [], [], [], [], []
    for topic, q, gold in QUERIES:
        top = retrieve(q, k=5, collection=collection)
        secs = [p.section for p in top]
        prec = sum(s in gold for s in secs[:3]) / 3
        rec = len(gold & set(secs)) / len(gold)
        p3.append(prec)
        r5.append(rec)
        hit3.append(float(any(s in gold for s in secs[:3])))
        p1.append(float(secs[0] in gold))
        rr.append(next((1 / (i + 1) for i, s in enumerate(secs) if s in gold), 0.0))
        ceil.append(min(len(gold), 3) / 3)
        rows.append(
            {
                "topic": topic,
                "query": q,
                "gold": sorted(gold),
                "top5": [p.chunk_id for p in top],
                "precision_at_3": round(prec, 3),
                "recall_at_5": round(rec, 3),
            }
        )
    n = len(QUERIES)
    topics = sorted({t for t, _, _ in QUERIES})
    by_topic = {}
    for t in topics:
        sel = [float(r["precision_at_3"]) for r in rows if r["topic"] == t]  # type: ignore[arg-type]
        by_topic[t] = round(sum(sel) / len(sel), 3)
    return {
        "collection": collection,
        "n_queries": n,
        "precision_at_3": round(sum(p3) / n, 4),
        "recall_at_5": round(sum(r5) / n, 4),
        "hit_at_3": round(sum(hit3) / n, 4),
        "precision_at_1": round(sum(p1) / n, 4),
        "mrr": round(sum(rr) / n, 4),
        # precision@3 cannot exceed min(|gold|, 3)/3 per query, so with these label sets the
        # attainable maximum is below 1; the ceiling-normalised value is reported next to it
        "precision_at_3_ceiling": round(sum(ceil) / n, 4),
        "precision_at_3_normalised": round(
            sum(a / b for a, b in zip(p3, ceil, strict=True)) / n, 4
        ),
        "precision_at_3_by_topic": by_topic,
        "queries": rows,
    }


def main() -> None:
    """Index the default chunking, benchmark it, and try alternatives if P@3 < 0.70."""
    results: dict[str, Any] = {}
    build_index(config_chunks("w150"), "kb_w150")
    results["w150"] = evaluate("kb_w150")
    best = "w150"
    if results["w150"]["precision_at_3"] < 0.70:
        for name in ("w100", "w200"):
            build_index(config_chunks(name), f"kb_{name}")
            results[name] = evaluate(f"kb_{name}")
        best = max(results, key=lambda k: results[k]["precision_at_3"])
    for k, v in results.items():
        print(
            k,
            "P@3",
            v["precision_at_3"],
            "R@5",
            v["recall_at_5"],
            "hit@3",
            v["hit_at_3"],
            "P@1",
            v["precision_at_1"],
            "MRR",
            v["mrr"],
            "ceiling",
            v["precision_at_3_ceiling"],
            "norm",
            v["precision_at_3_normalised"],
            v["precision_at_3_by_topic"],
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"selected": best, "configs": results}, indent=1), encoding="utf-8")
    print("selected:", best)


if __name__ == "__main__":
    main()
