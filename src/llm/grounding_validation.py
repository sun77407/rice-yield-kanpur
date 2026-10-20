"""Validate the grounding threshold on 10 handcrafted advisory samples (ADR-010, ADR-034).

Each sample is a set of KB passages plus sentences written by hand and labelled before any
similarity was computed: ``True`` = supported by those passages, ``False`` = plausible agronomy
that the passages do not support (some are deliberate near-misses on the same topic).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.llm.embed import embed
from src.llm.grounding import strip_citations
from src.rag.retriever import Passage

OUT = Path("kb/index/grounding_validation.json")
CHUNKS = Path("kb/chunks/kb.jsonl")

SAMPLES: list[tuple[list[str], list[tuple[str, bool]]]] = [
    (
        ["A1", "A2", "A4", "J1"],
        [
            (
                "The standard Kharif dose in eastern UP is 120 kg nitrogen, 60 kg P2O5 and 60 kg K2O per hectare, plus 5 kg zinc as zinc sulphate.",
                True,
            ),
            (
                "Apply half of the nitrogen with all the phosphorus, potash and zinc as basal dressing before transplanting.",
                True,
            ),
            (
                "Give the remaining nitrogen in two equal top-dressings at 40 and 60 days after transplanting.",
                True,
            ),
            (
                "On alkaline soils a foliar zinc sulphate spray at tillering and panicle initiation is more reliable than soil application.",
                True,
            ),
            ("Apply 200 kg of potash per hectare at flowering to improve grain filling.", False),
            ("Sulphur deficiency is the main reason for yellow leaves in Kanpur soils.", False),
        ],
    ),
    (
        ["B4", "B5", "B6", "B7"],
        [
            (
                "Keep about 3 cm of water after transplanting and raise it gradually to 5 to 10 cm.",
                True,
            ),
            (
                "Keep 5 cm of standing water from heading until the end of flowering, because water shortage then causes spikelet sterility.",
                True,
            ),
            ("Drain the field 7 to 10 days before harvest.", True),
            (
                "Alternate wetting and drying can save roughly 30 to 35% of irrigation water but should be paused during flowering.",
                True,
            ),
            ("Irrigate every 3 days with 10 cm of water throughout the season.", False),
            ("Apply gypsum at 500 kg per hectare to improve water holding.", False),
        ],
    ),
    (
        ["C1", "C2", "C3", "C4"],
        [
            (
                "Brown planthopper is the most damaging rice insect in Asia and can cause yield losses up to 60% in susceptible varieties.",
                True,
            ),
            (
                "Circular patches of dead yellow plants, known as hopper burn, mean the infestation is already severe, so scout weekly from panicle initiation.",
                True,
            ),
            (
                "Avoid excess urea and broad-spectrum insecticides such as cypermethrin, which can trigger planthopper outbreaks.",
                True,
            ),
            (
                "Resistant varieties such as IR64 and natural enemies like spiders help suppress the pest.",
                True,
            ),
            (
                "Spray imidacloprid at 0.5 ml per litre every 10 days as a preventive measure.",
                False,
            ),
            ("Release Trichogramma wasps at 50,000 per hectare to control planthoppers.", False),
        ],
    ),
    (
        ["D1", "D2", "D3", "D4", "D5"],
        [
            (
                "Blast shows as white to grey spindle-shaped lesions with dark borders and can break the stem at the neck during panicle development.",
                True,
            ),
            (
                "Humid weather, temperatures around 25 to 28 degrees and excess nitrogen favour blast.",
                True,
            ),
            (
                "Plant resistant varieties, remove crop residue and rotate fungicides with different modes of action.",
                True,
            ),
            (
                "Sheath blight begins as water-soaked greenish-grey lesions on lower leaves and no fully resistant cultivar is known.",
                True,
            ),
            ("Lower plant density and balanced nitrogen reduce sheath blight.", True),
            ("Apply tricyclazole at 0.6 g per litre at the boot leaf stage.", False),
            ("Blast cannot occur in alkaline soils.", False),
        ],
    ),
    (
        ["E4", "E5", "E6", "E7", "J2"],
        [
            (
                "Narendra-80 matures in 110 to 112 days, yields about 5.0 t/ha and is resistant to blast, tungro, sheath blight and stem borer.",
                True,
            ),
            (
                "If the monsoon is delayed beyond July 5, choose a short-duration variety of 90 to 110 days.",
                True,
            ),
            (
                "Salt-tolerant options for sodic soils are CSR-10, CSR-13, CSR-30 and the Narendra Usar varieties.",
                True,
            ),
            ("Pant Dhan-6 is a taller variety of 113 to 120 days averaging 4.1 t/ha.", True),
            ("Pusa 44 is the highest-yielding variety for Kanpur at 9 t/ha.", False),
            ("Hybrid rice gives double the yield of Narendra-80 under Kanpur conditions.", False),
        ],
    ),
    (
        ["F1", "F2", "F3", "F5", "I3"],
        [
            (
                "Sow the nursery from mid-May to early June and transplant 23 to 25 day old seedlings by the first week of July.",
                True,
            ),
            ("Flowers exposed to 35 degrees or more for over an hour can fail to set grain.", True),
            ("Treat seed with carbendazim or thiram at 2 g per kg before sowing.", True),
            ("Transplanted rice needs about 40 kg of seed per hectare in the nursery.", True),
            ("Use 150 kg of seed per hectare for a denser stand.", False),
            ("Soak seed in cold water for 3 days before sowing to break dormancy.", False),
        ],
    ),
    (
        ["J3", "J5", "J4", "I4"],
        [
            (
                "In a forecast bad year, register for PMFBY crop insurance before the cut-off date.",
                True,
            ),
            (
                "In a bad year the second nitrogen top-dressing at 60 days after transplanting becomes optional.",
                True,
            ),
            (
                "In a good year apply the full 120-60-60 NPK plus 5 kg zinc and watch for planthopper outbreaks.",
                True,
            ),
            ("The PMFBY premium for Kharif foodgrains is 1.5% of the sum insured.", True),
            ("Sell the entire crop to private traders before harvest to avoid price risk.", False),
            ("Switch the whole farm to sugarcane in a bad year.", False),
        ],
    ),
    (
        ["B1", "B2", "B3", "B8", "H5"],
        [
            (
                "Irrigated lowland rice typically needs 1,300 to 1,500 mm of water per season across Asia.",
                True,
            ),
            ("Bunds should be at least 20 cm high and sealed against cracks and rat holes.", True),
            (
                "Kanpur gets most of its rain between June and September, with normal monsoon onset around June 20 to 25.",
                True,
            ),
            (
                "A dry spell of 10 or more days in July and August is the most damaging rainfall anomaly and may need pump irrigation.",
                True,
            ),
            ("Install drip irrigation to cut water use by 70 percent in transplanted rice.", False),
            ("Rainfall in Kanpur has declined by 40% since 2000.", False),
        ],
    ),
    (
        ["A3", "A5", "A6", "H3", "H4", "I2"],
        [
            (
                "Soils of the Ganges plain near Kanpur are typically clay loam with pH 7.2 to 8.4, low in available nitrogen and high in potassium.",
                True,
            ),
            (
                "More than two thirds of applied phosphorus can become fixed, so residual phosphorus from the previous season counts.",
                True,
            ),
            (
                "Combining 5 to 10 tonnes of farmyard manure per hectare with chemical NPK keeps soil organic carbon up.",
                True,
            ),
            (
                "Half the recommended NPK with half of the nitrogen as farmyard manure plus zinc gave equivalent results in the NDUAT trial.",
                True,
            ),
            ("Potassium is deficient in Kanpur soils, so double the MOP dose.", False),
            ("Green manuring with Sesbania is required before every rice crop.", False),
        ],
    ),
    (
        ["G1", "G2", "C5", "H6"],
        [
            (
                "Harvest when grain moisture is about 20 to 25% and dry quickly to 14% or below.",
                True,
            ),
            (
                "Field losses can reach 10 to 20% and storage losses a further 5 to 15% under smallholder conditions.",
                True,
            ),
            ("Community-level rodent control works better than treating single plots.", True),
            (
                "Selling paddy to the mandi at MSP depends on moisture content and procurement timing.",
                True,
            ),
            ("Store paddy at 20% moisture in jute bags for up to a year.", False),
            ("Harvest at 35% moisture to reduce shattering.", False),
        ],
    ),
]


def load_passages() -> dict[str, Passage]:
    """KB chunks by id as ``Passage`` objects."""
    out = {}
    for line in CHUNKS.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        out[c["chunk_id"]] = Passage(
            c["chunk_id"], c["source_section"], c["title"], c["text"], "", c["source_type"], 0.0
        )
    return out


def sweep() -> dict[str, object]:
    """Threshold sweep: balanced accuracy, and precision/recall of the ``grounded`` call."""
    kb = load_passages()
    maxsim: list[float] = []
    label: list[bool] = []
    for ids, sents in SAMPLES:
        pv = embed([f"{kb[i].title}. {kb[i].text}" for i in ids])
        sv = embed([strip_citations(s) for s, _ in sents])
        maxsim.extend((sv @ pv.T).max(axis=1).tolist())
        label.extend(g for _, g in sents)
    ms, lab = np.array(maxsim), np.array(label)
    rows = []
    for t in np.round(np.arange(0.40, 0.91, 0.02), 2):
        pred = ms >= t
        tp = int((pred & lab).sum())
        fp = int((pred & ~lab).sum())
        fn = int((~pred & lab).sum())
        tn = int((~pred & ~lab).sum())
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        rows.append(
            {
                "threshold": float(t),
                "grounded_recall": round(tpr, 3),  # supported sentences correctly kept
                "unsupported_rejected": round(tnr, 3),
                "balanced_accuracy": round((tpr + tnr) / 2, 3),
                "false_negatives": fn,  # supported but flagged as ungrounded
                "false_positives": fp,  # unsupported but passed
            }
        )
    best = max(rows, key=lambda r: (r["balanced_accuracy"], -abs(r["threshold"] - 0.6)))
    return {
        "n_sentences": len(lab),
        "n_supported": int(lab.sum()),
        "max_sim_supported": {
            "min": float(ms[lab].min()),
            "median": float(np.median(ms[lab])),
            "max": float(ms[lab].max()),
        },
        "max_sim_unsupported": {
            "min": float(ms[~lab].min()),
            "median": float(np.median(ms[~lab])),
            "max": float(ms[~lab].max()),
        },
        "sweep": rows,
        "best_by_balanced_accuracy": best,
    }


def main() -> None:
    """Run the sweep and write the result."""
    res = sweep()
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print({k: v for k, v in res.items() if k != "sweep"})
    for r in res["sweep"]:  # type: ignore[attr-defined]
        if r["threshold"] in (0.5, 0.56, 0.6, 0.64, 0.68, 0.7, 0.74, 0.78):
            print(r)


if __name__ == "__main__":
    main()
